// Head tracking with the webcam: the perspective follows your head. MediaPipe Face Landmarker
// runs on-device in a worker (headtrack.worker.js); no video leaves the browser.
//
// Everything is in real centimetres, so the layers move by a physically consistent amount for
// your screen and your distance. The ruler is your pupil distance (~6.3 cm), which turns the
// camera image into centimetres without knowing the camera's field of view.
//
// Latency is what breaks the illusion, so the pipeline keeps it short and makes it measurable:
// the newest camera frame goes to the worker as soon as it's free, the One Euro filter and the
// prediction are tunable live, and measureRoundTrip() times screen → camera with light flashes.

const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0';
const MODEL = new URL('../assets/models/face_landmarker.task', import.meta.url).href;

export const IPD_CM = 6.3;     // average adult pupil distance
const FOV_DEG = 60;            // across the long side of the camera image; only used for your distance
const LOST_AFTER_MS = 1000;
const MAX_LEAD_S = 0.25;

/** A phone held in the hand: closer to your face, denser pixels, upright camera image. */
export const PHONE = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 600;

/**
 * Your distance to the camera (cm) from the pupil distance in the image (in image widths).
 * aspect = image height / width; a phone held upright gives a portrait image, whose width is the
 * short side and sees a narrower angle.
 */
export function distanceFromPupils(pupils, aspect) {
  const tanLong = Math.tan((FOV_DEG * Math.PI) / 360);
  const tanWidth = aspect > 1 ? tanLong / aspect : tanLong;
  return IPD_CM / (pupils * 2 * tanWidth);
}

/**
 * In centimetres. minCutoff (Hz): steadiness when still · beta: how fast it opens up when you
 * move · predictMs: extra lead beyond "now" for camera and screen latency (the time since the
 * frame was captured is always extrapolated) · depthCm: how deep the relief is · delegate: where
 * the face model runs ('auto' times GPU and CPU and keeps the faster).
 */
export const DEFAULT_TUNING = { minCutoff: 0.8, beta: 0.4, dCutoff: 4, predictMs: 30, depthCm: 10, delegate: 'auto' };
/**
 * center: where your eyes are in the camera image (0–1) when you look straight at the screen ·
 * pupils: your pupil distance in the image (0 = estimate on the fly) · rangeCm: how far you move
 * · pxPerCm: CSS pixels per centimetre on this screen (CSS assumes 96 per inch; phones have ~150).
 */
export const DEFAULT_CALIBRATION = { center: [0.5, 0.5], pupils: 0, rangeCm: PHONE ? 5 : 8, pxPerCm: PHONE ? 60 : 96 / 2.54 };

export class HeadTracker {
  active = false;
  hasFace = false;
  tuning = { ...DEFAULT_TUNING };
  calibration = structuredClone(DEFAULT_CALIBRATION);
  /**
   * cameraFps · trackMs (inference) · pipelineMs (capture → result) · captureLagMs (capture → browser) ·
   * delegate ('GPU', 'CPU' or 'auto' while timing both) · delegateMs (what each took when timed)
   */
  stats = { cameraFps: 0, trackFps: 0, trackMs: 0, pipelineMs: 0, captureLagMs: 0, delegate: '', delegateMs: null };
  /** Latest unfiltered head offset (cm) and its capture time (for the tuning graph) */
  raw = { x: 0, y: 0, t: 0 };
  #autoPupils = 0;

  #worker = null;
  #stream = null;
  #video = null;
  #dot = null;
  #busy = false;
  #latest = null;        // newest captured frame waiting for the worker
  #lastSeen = 0;
  #lastResult = 0;
  #lastCameraFrame = 0;
  #sampleTime = 0;
  #listeners = new Set();
  #axes = [new OneEuro(this.tuning), new OneEuro(this.tuning)];

  /** @param {HTMLVideoElement} video  @param {HTMLElement} dot  marker drawn over the preview */
  async start(video, dot) {
    if (this.active) return;
    if (!window.isSecureContext) throw new Error('Kameraet krever https eller localhost.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Nettleseren har ikke tilgang til kamera.');

    const worker = new Worker(new URL('./headtrack.worker.js', import.meta.url), { type: 'module' });
    const ready = new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => (data.type === 'ready' ? resolve(data) : reject(new Error(data.message)));
      worker.onerror = (e) => reject(new Error(e.message || 'Ansiktssporingen kunne ikke starte'));
    });
    worker.postMessage({ type: 'init', base: MP_BASE, model: MODEL, delegate: this.tuning.delegate ?? 'auto' });

    try {
      const [stream, info] = await Promise.all([
        navigator.mediaDevices.getUserMedia({
          // Ask for 60 fps: every camera frame interval is latency
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
          audio: false,
        }),
        ready,
      ]);
      this.stats.delegate = info.delegate;
      this.stats.delegateMs = null;
      this.#stream = stream;
      video.srcObject = stream;
      await video.play();
    } catch (err) {
      worker.terminate();
      this.#stream?.getTracks().forEach((t) => t.stop());
      this.#stream = null;
      throw err;
    }

    worker.onmessage = ({ data }) => {
      if (data.type === 'result') this.#onResult(data);
      else if (data.type === 'delegate') {
        this.stats.delegate = data.delegate;
        this.stats.delegateMs = data.ms;
      }
    };
    this.#worker = worker;
    this.#video = video;
    this.#dot = dot;
    this.#busy = false;
    this.active = true;
    this.#nextFrame();
  }

  stop() {
    this.active = false;
    this.hasFace = false;
    this.#latest?.frame.close();
    this.#latest = null;
    this.#worker?.postMessage({ type: 'close' });
    this.#worker = null;
    this.#stream?.getTracks().forEach((t) => t.stop());
    this.#stream = null;
    if (this.#video) this.#video.srcObject = null;
  }

  /**
   * Head offset in cm from where you sat during calibration (+x = to your right, +y = down).
   * Extrapolated from the frame's capture time to `now`, plus `predictMs` for the latency
   * before capture and on the screen.
   */
  offsetAt(now) {
    const lead = Math.min(MAX_LEAD_S, Math.max(0, (now - this.#sampleTime + this.tuning.predictMs) / 1000));
    return this.#axes.map((f) => (f.x ?? 0) + f.dx * lead);
  }

  /** Your distance to the screen in cm (from the pupil distance and an assumed camera field of view). */
  get distanceCm() {
    return Math.min(200, Math.max(20, distanceFromPupils(this.#pupils(), this.aspect)));
  }

  /** Height/width of the camera image. */
  get aspect() {
    return this.#video ? this.#video.videoHeight / this.#video.videoWidth : 0.75;
  }

  #pupils() {
    return this.calibration.pupils || this.#autoPupils || 0.1;
  }

  /** Raw camera-image samples ({cx, cy, pupils, t}) seen during the next `ms` milliseconds. */
  collect(ms) {
    return new Promise((resolve) => {
      const samples = [];
      const add = (s) => samples.push(s);
      this.#listeners.add(add);
      setTimeout(() => {
        this.#listeners.delete(add);
        resolve(samples);
      }, ms);
    });
  }

  /**
   * Screen → camera → browser latency: flash the screen and time how long until the camera
   * sees your face light up. Needs your face in front of the screen; a dim room works best.
   * @param {HTMLElement} flash  full-screen element to turn black/white
   * @returns {Promise<{median: number, samples: number[]}>} milliseconds
   */
  async measureRoundTrip(flash, rounds = 8) {
    if (!this.active) throw new Error('Slå på «Følg hodet» først.');
    const v = this.#video;
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 24;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const brightness = () => {
      ctx.drawImage(v, 0, 0, 32, 24);
      const d = ctx.getImageData(8, 6, 16, 12).data; // centre, where your face is
      let sum = 0;
      for (let i = 0; i < d.length; i += 4) sum += d[i] + d[i + 1] + d[i + 2];
      return sum / (d.length / 4) / 3;
    };
    const frames = (ms) =>
      new Promise((resolve) => {
        const out = [];
        const end = performance.now() + ms;
        const step = () => {
          out.push({ t: performance.now(), b: brightness() });
          if (performance.now() < end) v.requestVideoFrameCallback(step);
          else resolve(out);
        };
        v.requestVideoFrameCallback(step);
      });
    const paintAt = (color) =>
      new Promise((resolve) =>
        requestAnimationFrame((t) => {
          flash.style.background = color;
          resolve(t); // the colour reaches the screen on the frame after this one
        }),
      );

    flash.hidden = false;
    const samples = [];
    try {
      for (let i = 0; i < rounds; i++) {
        await paintAt('#000');
        const dark = await frames(500);
        const base = dark.slice(-4).reduce((s, f) => s + f.b, 0) / 4;
        const lit = await paintAt('#fff');
        const after = await frames(600);
        const peak = Math.max(...after.map((f) => f.b));
        if (peak - base < 4) continue; // the camera didn't see it (bright room, face too far away)
        const hit = after.find((f) => f.b >= base + (peak - base) / 2);
        samples.push(hit.t - lit);
      }
    } finally {
      flash.hidden = true;
      flash.style.background = '#000';
    }
    if (!samples.length) throw new Error('Kameraet så ikke blinkene. Prøv i et mørkere rom, nærmere skjermen.');
    samples.sort((a, b) => a - b);
    return { median: samples[samples.length >> 1], samples };
  }

  #nextFrame() {
    const v = this.#video;
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(this.#onVideoFrame);
    else requestAnimationFrame((now) => this.#onVideoFrame(now, {}));
  }

  #onVideoFrame = (now, metadata) => {
    if (!this.active) return;
    this.#nextFrame();
    if (this.#lastCameraFrame) this.stats.cameraFps += (1000 / (now - this.#lastCameraFrame) - this.stats.cameraFps) * 0.1;
    this.#lastCameraFrame = now;
    const t = metadata.captureTime ?? now;
    this.stats.captureLagMs += (performance.now() - t - this.stats.captureLagMs) * 0.1;
    // Grab every frame, but only the newest waits for the worker (a mailbox, not a queue):
    // when tracking finishes, the next frame is already there instead of up to a frame away
    createImageBitmap(this.#video)
      .then((frame) => {
        this.#latest?.frame.close();
        this.#latest = { frame, t };
        this.#pump();
      })
      .catch(() => {});
  };

  #pump() {
    if (this.#busy || !this.#latest || !this.#worker) return;
    const { frame, t } = this.#latest;
    this.#latest = null;
    this.#busy = true;
    this.#worker.postMessage({ type: 'frame', frame, t }, [frame]);
  }

  #onResult({ t, ms, anchor, pupils }) {
    this.#busy = false;
    this.#pump();
    if (!this.active) return;
    const now = performance.now();
    this.stats.trackMs += (ms - this.stats.trackMs) * 0.2;
    this.stats.pipelineMs += (now - t - this.stats.pipelineMs) * 0.2;
    if (this.#lastResult) this.stats.trackFps += (1000 / (now - this.#lastResult) - this.stats.trackFps) * 0.2;
    this.#lastResult = now;

    if (anchor && pupils) {
      const aspect = this.aspect;
      const span = Math.hypot(pupils.dx, pupils.dy * aspect); // pupil distance in image widths
      // Uncalibrated: follow your pupil distance slowly (calibration fixes it)
      this.#autoPupils = this.#autoPupils ? this.#autoPupils + (span - this.#autoPupils) * 0.05 : span;
      this.#listeners.forEach((add) => add({ cx: anchor.x, cy: anchor.y, pupils: span, t }));

      // Image → centimetres with the pupils as ruler. The camera faces you, so your right is its left.
      const cmPerWidth = IPD_CM / this.#pupils();
      const [x0, y0] = this.calibration.center;
      const offset = [(x0 - anchor.x) * cmPerWidth, (anchor.y - y0) * aspect * cmPerWidth];
      this.raw = { x: offset[0], y: offset[1], t };
      if (!this.hasFace) this.#axes.forEach((f) => f.reset()); // reappearing: start where you are
      offset.forEach((value, i) => this.#axes[i].filter(value, t / 1000));
      this.#sampleTime = t;
      this.hasFace = true;
      this.#lastSeen = now;
      // The preview is mirrored like a selfie, so mirror the marker too
      this.#dot.style.left = `${(1 - anchor.x) * 100}%`;
      this.#dot.style.top = `${anchor.y * 100}%`;
    } else if (now - this.#lastSeen > LOST_AFTER_MS) {
      this.hasFace = false;
    }
    this.#dot.hidden = !this.hasFace;
  }
}

/** One Euro filter (Casiez et al. 2012): low jitter at low speed, low lag at high speed. Reads its
 *  parameters from a shared object so they can be tuned while it runs. */
class OneEuro {
  constructor(params) {
    this.params = params;
    this.reset();
  }

  reset() {
    this.x = null;
    this.dx = 0;
    this.t = 0;
  }

  filter(x, t) {
    if (this.x === null || t <= this.t) {
      this.x = x;
      this.dx = 0;
      this.t = t;
      return x;
    }
    const { minCutoff, beta, dCutoff } = this.params;
    const dt = Math.min(0.25, t - this.t);
    this.t = t;
    this.dx += ((x - this.x) / dt - this.dx) * alpha(dCutoff, dt);
    this.x += (x - this.x) * alpha(minCutoff + beta * Math.abs(this.dx), dt);
    return this.x;
  }
}

function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}
