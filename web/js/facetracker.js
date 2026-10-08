// Face tracking with MediaPipe Face Landmarker, driven by messages so it can run in a worker
// (headtrack.worker.js, the normal case: it can't stall rendering) or on the main thread (the
// fallback where MediaPipe doesn't work in a worker, see headtrack.js). In video mode it tracks
// the face from the previous frame instead of detecting it anew each time, which is what keeps
// the position from jumping.
//
// GPU or CPU: which is faster depends on the machine. On a laptop where the painting and the
// face model share an integrated GPU, the CPU can win by a wide margin. With delegate 'auto' both
// run on alternate camera frames until each has timed enough frames with a face, then the
// faster one stays.

// Stable point between the eyes: inner eye corners weighted more than the irises (as in
// jasondecamp/desktop-vr). The corners don't move when you blink or look around.
const ANCHOR = [[133, 0.3], [362, 0.3], [468, 0.2], [473, 0.2]];
const TRIAL_FRAMES = 15;  // timed frames with a face per delegate before choosing
const TRIAL_WARMUP = 3;   // the first calls compile shaders and allocate: not representative
const TRIAL_MAX = 240;    // frames; choose from what we have if no face shows up for long

const IN_WORKER = typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;
const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];

function create(FaceLandmarker, fileset, model, delegate) {
  // The wasm loader hands over its module once per load; a URL of its own per delegate makes
  // it run again, otherwise the second landmarker fails with "ModuleFactory not set"
  const own = { ...fileset, wasmLoaderPath: `${fileset.wasmLoaderPath}?${delegate}` };
  return FaceLandmarker.createFromOptions(own, {
    // Our own canvas: MediaPipe only trusts OffscreenCanvas in browsers whose user agent says
    // Safari ≥ 17 or Chrome, and otherwise makes one with document.createElement. Chrome, Arc
    // and in-app browsers on iPhone say neither, and a worker has no document.
    canvas: typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas'),
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

/**
 * @param {(message: object) => void} post  where results go
 * @returns {(message: object) => Promise<void>}  handles init, frame and close messages
 */
export function createFaceTracker(post) {
  let candidates = []; // [{ delegate, landmarker, times, calls }]; one left once chosen
  let turn = 0;
  let frames = 0;

  /** Keep the faster delegate once both have been timed (on frames with a face, so like for like). */
  function maybeChoose() {
    if (candidates.length < 2) return;
    const enough = candidates.every((c) => c.times.length >= TRIAL_FRAMES);
    if (!enough && frames < TRIAL_MAX) return;
    const ms = candidates.map((c) => (c.times.length ? median(c.times) : Infinity));
    const best = candidates[ms.indexOf(Math.min(...ms))];
    candidates.filter((c) => c !== best).forEach((c) => c.landmarker.close());
    const timings = Object.fromEntries(candidates.map((c, i) => [c.delegate, Number.isFinite(ms[i]) ? Math.round(ms[i]) : null]));
    candidates = [best];
    post({ type: 'delegate', delegate: best.delegate, ms: timings });
  }

  async function init(msg) {
    try {
      if (IN_WORKER && typeof OffscreenCanvas === 'undefined') {
        throw new Error('Nettleseren er for gammel for hodesporing (iOS 16.4 eller nyere trengs).');
      }
      const { FilesetResolver, FaceLandmarker } = await import(`${msg.base}/vision_bundle.mjs`);
      // Module workers load the ES module build; the main thread the classic one, via <script>
      const fileset = await FilesetResolver.forVisionTasks(`${msg.base}/wasm`, IN_WORKER);
      const wanted = msg.delegate === 'auto' ? ['GPU', 'CPU'] : [msg.delegate];
      let firstError = null;
      for (const delegate of wanted) {
        try {
          candidates.push({ delegate, landmarker: await create(FaceLandmarker, fileset, msg.model, delegate), times: [], calls: 0 });
        } catch (err) {
          firstError ??= err; // no WebGL2 here, for example: try the next one
        }
      }
      if (!candidates.length) throw firstError;
      post({ type: 'ready', delegate: candidates.length > 1 ? 'auto' : candidates[0].delegate });
    } catch (err) {
      post({ type: 'error', message: String(err?.message ?? err) });
    }
  }

  function frame(msg) {
    // While choosing, alternate frames between the delegates; each tracks with its own state
    const c = candidates[turn++ % candidates.length];
    frames++;
    const started = performance.now();
    const face = c.landmarker.detectForVideo(msg.frame, msg.t).faceLandmarks[0];
    const ms = performance.now() - started;
    msg.frame.close();
    if (candidates.length > 1) {
      if (face && ++c.calls > TRIAL_WARMUP) c.times.push(ms);
      maybeChoose();
    }
    let anchor = null;
    let pupils = null;
    if (face) {
      anchor = { x: 0, y: 0 };
      for (const [i, w] of ANCHOR) {
        anchor.x += face[i].x * w;
        anchor.y += face[i].y * w;
      }
      // Pupil to pupil (normalised x and y): ~6.3 cm on an adult, our ruler for centimetres
      pupils = { dx: face[473].x - face[468].x, dy: face[473].y - face[468].y };
    }
    post({ type: 'result', t: msg.t, ms, anchor, pupils });
  }

  return async (msg) => {
    if (msg.type === 'init') await init(msg);
    else if (msg.type === 'frame') frame(msg);
    else if (msg.type === 'close') {
      candidates.forEach((c) => c.landmarker.close());
      candidates = [];
    }
  };
}
