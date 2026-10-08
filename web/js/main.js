import { ParallaxRenderer } from './renderer.js';
import { TiltInput } from './input.js';
import { HeadTracker, DEFAULT_TUNING, DEFAULT_CALIBRATION } from './headtrack.js';
import { openCalibration, closeCalibration } from './calibration.js';

const params = new URLSearchParams(location.search);
const EMBED = params.has('embed');   // hero image on a page: fills its box, no UI
const KIOSK = params.has('kiosk');   // museum screen: no UI, cycles through the works
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;

const KIOSK_INTERVAL_MS = 25_000;
const CLIP_SECONDS = 6;
const CLIP_FORMATS = { '4:5': [1080, 1350], '9:16': [1080, 1920], '1:1': [1080, 1080], '16:9': [1920, 1080] };
const MODE_HINTS = {
  1: 'Følger synslinjen gjennom dybdekartet, så forgrunnen dekker bakgrunnen riktig.',
  0: 'uv += (dybde − fokus) · tilt · styrke. Billig, men kantene smøres ut.',
};

const $ = (id) => document.getElementById(id);
const canvas = $('stage');
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const approach = (cur, target, dt, rate) => cur + (target - cur) * (1 - Math.exp(-dt * rate));

const state = {
  works: [],
  index: -1,
  mode: 1,
  strength: REDUCED_MOTION ? 0.012 : 0.025,  // fraction of painting width at full tilt
  focus: 0.5,
  focusTarget: 0.5,
  maxSteps: 64,
  breathe: (EMBED || KIOSK) && !REDUCED_MOTION,
  dust: false,
  dolly: false,
  showDepth: false,
  original: false,
  head: false,
  // contain: inside the layout next to the UI · fill: as large as possible on the whole screen, never cropped
  fit: EMBED || KIOSK ? 'fill' : 'contain',
  clipFormat: '4:5',
  recording: null,
};

// Smoothed values driven towards the state every frame
const anim = { tilt: [0, 0], breath: 0, dust: 0, dolly: 0, head: 0, headCm: [0, 0], showDepth: 0, fade: 0, focusFlash: 0, effect: 1 };
let fadeTween = null;
let current = null;     // { image, depth, depthData } for the shown work
let lastRect = null;
let renderer;
let input;
const head = new HeadTracker();

const logo = new Image();
logo.src = 'assets/logo/museet-logo-nam--white.svg';

// ---------- Loading ----------

const cache = new Map();

async function bitmap(url, raw = false) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} (${res.status})`);
  const blob = await res.blob();
  // Depth must reach the shader untouched: no colour management or premultiplication
  return createImageBitmap(blob, raw ? { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } : {});
}

function readDepth(bmp) {
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  return { w: c.width, h: c.height, data: ctx.getImageData(0, 0, c.width, c.height).data };
}

/**
 * Depth of the background: sky and distant mountains, or the back wall. A low percentile of the
 * depth map, so a few stray pixels don't decide it.
 */
function backgroundDepth({ data }) {
  const values = [];
  for (let i = 0; i < data.length; i += 4 * 7) values.push(data[i]);
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length * 0.05)] / 255;
}

function loadWork(work) {
  if (!cache.has(work.slug)) {
    const p = Promise.all([bitmap(work.image), bitmap(work.depth, true)]).then(([image, depth]) => {
      const depthData = readDepth(depth);
      return { image, depth, depthData, background: backgroundDepth(depthData) };
    });
    p.catch(() => cache.delete(work.slug));
    cache.set(work.slug, p);
  }
  return cache.get(work.slug);
}

function fadeTo(value, ms) {
  fadeTween?.resolve(); // a newer fade replaces an unfinished one
  return new Promise((resolve) => {
    fadeTween = { from: anim.fade, to: value, start: performance.now(), ms, resolve };
    // The frame loop finishes the tween, but don't hang if frames are paused (background tab)
    setTimeout(resolve, ms + 100);
  });
}

let kioskTimer = null;
function scheduleKiosk() {
  clearInterval(kioskTimer);
  kioskTimer = setInterval(() => step(1), KIOSK_INTERVAL_MS);
}

// Count from the last requested work, so quick presses don't stall while an image is loading
let requested = -1;
const step = (delta) => show((requested >= 0 ? requested : state.index) + delta);

let showToken = 0;
async function show(index) {
  const n = state.works.length;
  index = ((index % n) + n) % n;
  requested = index;
  const token = ++showToken;
  const work = state.works[index];
  const [assets] = await Promise.all([loadWork(work), state.index >= 0 ? fadeTo(0, 180) : null]).catch((e) => {
    fail(`Fant ikke bildet eller dybdekartet for «${work.title}»`, e);
    return [];
  });
  if (!assets || token !== showToken) return;

  renderer.setPainting(assets.image, assets.depth);
  current = assets;
  state.index = index;
  state.focus = state.focusTarget = defaultFocus(work);
  renderCaption(work);
  syncPanel();
  if (!EMBED && !KIOSK) history.replaceState(null, '', `?work=${work.slug}`);
  if (KIOSK) scheduleKiosk(); // a manual change restarts the countdown
  fadeTo(1, 450);

  loadWork(state.works[(index + 1) % n]);
  loadWork(state.works[(index - 1 + n) % n]);
}

// ---------- Frame loop ----------

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const [w, h] = state.recording
    ? state.recording.size
    : [Math.round(canvas.clientWidth * dpr), Math.round(canvas.clientHeight * dpr)];
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return dpr;
}

// DOM measurements are cached and refreshed on resize: reading layout in the frame loop right
// after writing styles forces a synchronous layout every frame, which shows up as hitches.
const measured = { area: { left: 0, top: 0, width: 1, height: 1 }, badgeWidth: 0 };
function measureLayout() {
  const r = $('frame-area').getBoundingClientRect();
  measured.area = { left: r.left, top: r.top, width: r.width, height: r.height };
  measured.badgeWidth = $('ai-badge').offsetWidth;
}

function frameArea(dpr) {
  const r = measured.area;
  return { x: r.left * dpr, y: r.top * dpr, w: r.width * dpr, h: r.height * dpr };
}

let last = performance.now();
let fps = 60;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (dt > 0) fps = approach(fps, 1 / dt, dt, 2);
  const t = now / 1000;
  const dpr = resize();
  const work = state.works[state.index];
  if (!work) return;

  if (fadeTween) {
    const k = clamp((now - fadeTween.start) / fadeTween.ms, 0, 1);
    anim.fade = fadeTween.from + (fadeTween.to - fadeTween.from) * (k * k * (3 - 2 * k));
    if (k === 1) {
      fadeTween.resolve();
      fadeTween = null;
    }
  }

  const rec = state.recording;
  const strength = state.strength * (work.strength ?? 1);
  const fit = rec ? 'cover' : state.fit;
  const rect = renderer.layout(fit, frameArea(dpr));
  lastRect = { ...rect, dpr };

  let shift;
  let dolly;
  let focus = state.focus;
  let shiftRange = 1.2 * strength;
  let dollyRange = Math.max(anim.dolly * 0.22, 0.03);
  state.focus = approach(state.focus, state.focusTarget, dt, 5);

  if (rec) {
    // A closed loop around the painting, so the clip can repeat seamlessly
    const a = clamp((now - rec.start) / rec.ms, 0, 1) * Math.PI * 2;
    anim.tilt = [0.9 * Math.cos(a), 0.5 * Math.sin(a)];
    shift = [anim.tilt[0] * strength, anim.tilt[1] * strength];
    dolly = state.dolly ? 0.18 * Math.sin(a) : 0;
    anim.effect = 1;
  } else {
    // Mouse, gyro and breathing orbit the camera around the focus plane
    const idle = now - input.lastActivity > 2500;
    anim.tilt[0] = approach(anim.tilt[0], input.target[0], dt, 6);
    anim.tilt[1] = approach(anim.tilt[1], input.target[1], dt, 6);
    anim.breath = approach(anim.breath, state.breathe ? (idle ? 1 : 0.25) : 0, dt, 1.5);
    anim.dolly = approach(anim.dolly, state.dolly ? 1 : 0, dt, 2);
    anim.effect = approach(anim.effect, state.original ? 0 : 1, dt, 5);
    const orbit = [0.55 * Math.sin(t * 0.43), 0.32 * Math.sin(t * 0.61 + 1.3)]; // "breathing"
    const tilt = [
      clamp(anim.tilt[0] + orbit[0] * anim.breath, -1.2, 1.2),
      clamp(anim.tilt[1] + orbit[1] * anim.breath, -1.2, 1.2),
    ];
    const orbitShift = [tilt[0] * strength, tilt[1] * strength];
    const orbitDolly = anim.dolly * 0.22 * Math.sin(t * 0.7) + anim.breath * 0.025 * Math.sin(t * 0.31);

    // A visible face takes over (see headView). Camera frames come at ~30 Hz; a short ease
    // turns them into continuous motion at the display rate.
    anim.head = approach(anim.head, head.active && head.hasFace ? 1 : 0, dt, 4);
    const cm = head.offsetAt(now);
    anim.headCm[0] = approach(anim.headCm[0], cm[0], dt, 50);
    anim.headCm[1] = approach(anim.headCm[1], cm[1], dt, 50);
    if (head.active) recordHeadSamples(now);
    const m = anim.head;
    const view = m > 0.001 ? headView(rect, dpr) : { shift: [0, 0], range: 0 };
    const mix = (a, b) => a + (b - a) * m;
    shift = [mix(orbitShift[0], view.shift[0]) * anim.effect, mix(orbitShift[1], view.shift[1]) * anim.effect];
    dolly = mix(orbitDolly, 0) * anim.effect;
    shiftRange = mix(shiftRange, view.range);
  }
  anim.dust = approach(anim.dust, state.dust ? 1 : 0, dt, 2);
  anim.showDepth = approach(anim.showDepth, state.showDepth ? 1 : 0, dt, 6);
  anim.focusFlash = Math.max(0, anim.focusFlash - dt * 0.8);

  const info = renderer.render({
    rect,
    shift,
    focus,
    dolly,
    // Scaling the ranges by the effect lets the overscan relax, so "original" shows the full canvas
    shiftRange: shiftRange * anim.effect,
    dollyRange: dollyRange * anim.effect,
    mode: state.mode,
    maxSteps: state.maxSteps,
    showDepth: anim.showDepth * anim.effect,
    focusFlash: Math.min(1, anim.focusFlash) * anim.effect,
    fade: anim.fade,
    dust: anim.dust * anim.effect,
    time: t,
  });

  if (rec) rec.compose(work);
  placeBadge(rect, dpr);
  if (Math.floor(now / 250) !== Math.floor((now - dt * 1000) / 250) && info) {
    $('fps').textContent = fps.toFixed(0);
    $('steps-now').textContent = state.mode ? info.steps : '1';
    if (head.active) showHeadStats();
  }
  if (head.active && !$('panel').hidden) drawHeadGraph(now);
}

// ---------- Head tracking: measure and tune latency ----------

// v2: centimetres (earlier versions stored angles, which don't carry over)
const TUNING_KEY = 'levende-bilder:head-tuning-v2';
const CALIBRATION_KEY = 'levende-bilder:head-calibration-v2';
let calibratedAt = null;
const GRAPH_MS = 2500;
const headSamples = { raw: [], out: [], lag: null, lastLagAt: 0, roundTrip: null };

function recordHeadSamples(now) {
  const { raw, out } = headSamples;
  if (head.hasFace && head.raw.t !== raw.at(-1)?.t) raw.push({ t: head.raw.t, x: head.raw.x }); // by capture time
  out.push({ t: now, x: anim.headCm[0] }); // by display time
  while (raw.length && raw[0].t < now - GRAPH_MS) raw.shift();
  while (out.length && out[0].t < now - GRAPH_MS) out.shift();
  if (now - headSamples.lastLagAt > 500) {
    headSamples.lastLagAt = now;
    headSamples.lag = estimateLag(raw, out) ?? headSamples.lag;
  }
}

/** Value of the sampled curve at time t (linear interpolation), or null outside it. */
function sampleAt(samples, t) {
  if (!samples.length || t < samples[0].t || t > samples.at(-1).t) return null;
  let i = 1;
  while (samples[i].t < t) i++;
  const a = samples[i - 1];
  const b = samples[i];
  return a.x + ((b.x - a.x) * (t - a.t)) / Math.max(1e-3, b.t - a.t);
}

/**
 * How far behind (ms) what we draw is compared to where the camera saw your head: the time
 * shift that best lines up the two curves. Needs some movement to measure anything.
 * Negative = the prediction runs ahead.
 */
function estimateLag(raw, out) {
  if (raw.length < 10) return null;
  const xs = raw.map((s) => s.x);
  if (Math.max(...xs) - Math.min(...xs) < 2) return null; // cm: hold still and there's nothing to compare
  let best = null;
  let bestErr = Infinity;
  for (let lag = -100; lag <= 300; lag += 5) {
    let err = 0;
    let n = 0;
    for (const o of out) {
      const r = sampleAt(raw, o.t - lag);
      if (r === null) continue;
      err += (o.x - r) ** 2;
      n++;
    }
    if (n > 30 && err / n < bestErr) {
      bestErr = err / n;
      best = lag;
    }
  }
  return best;
}

function drawHeadGraph(now) {
  if ($('head-tuning').hidden) return;
  const canvas = $('head-graph');
  const ctx = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  const all = [...headSamples.raw, ...headSamples.out].map((s) => Math.abs(s.x));
  const scale = Math.max(2, ...all) * 1.1; // cm
  const toX = (t) => ((t - (now - GRAPH_MS)) / GRAPH_MS) * w;
  const toY = (x) => h / 2 - (x / scale) * (h / 2);
  ctx.strokeStyle = '#323232';
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  const line = (samples, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    samples.forEach((s, i) => (i ? ctx.lineTo(toX(s.t), toY(s.x)) : ctx.moveTo(toX(s.t), toY(s.x))));
    ctx.stroke();
  };
  line(headSamples.raw, '#969696', 1.5);
  line(headSamples.out, '#ffcb05', 2);
}

function showHeadStats() {
  const s = head.stats;
  const lag = headSamples.lag;
  $('t-camera').textContent = `${s.cameraFps.toFixed(0)} b/s`;
  const other = Object.entries(s.delegateMs ?? {}).filter(([d, ms]) => d !== s.delegate && ms !== null);
  const how =
    s.delegate === 'auto'
      ? 'måler GPU og CPU …'
      : s.delegate + other.map(([d, ms]) => `; ${d} tok ${ms} ms`).join('');
  $('t-track').textContent = `${s.trackMs.toFixed(0)} ms · ${s.trackFps.toFixed(0)} b/s (${how})`;
  $('t-pipeline').textContent = `${s.pipelineMs.toFixed(0)} ms`;
  $('t-lag').textContent = lag === null ? 'beveg hodet sidelengs' : `${lag} ms`;
  const rt = headSamples.roundTrip;
  $('t-total').textContent =
    rt === null ? 'mål forsinkelse først' : `≈ ${Math.round(rt + s.trackMs + Math.max(0, lag ?? 0))} ms`;
}

function loadTuning() {
  try {
    Object.assign(head.tuning, JSON.parse(localStorage.getItem(TUNING_KEY) || '{}'));
    const saved = JSON.parse(localStorage.getItem(CALIBRATION_KEY) || 'null');
    if (saved?.center && saved.pxPerCm) {
      const { at, ...calibration } = saved;
      head.calibration = { ...structuredClone(DEFAULT_CALIBRATION), ...calibration };
      calibratedAt = at;
    }
  } catch {
    // private mode or garbage: defaults are fine
  }
}

function saveTuning() {
  try {
    localStorage.setItem(TUNING_KEY, JSON.stringify(head.tuning));
    if (calibratedAt) localStorage.setItem(CALIBRATION_KEY, JSON.stringify({ ...head.calibration, at: calibratedAt }));
    else localStorage.removeItem(CALIBRATION_KEY);
  } catch {
    // not essential
  }
}

/** Calibration finished: apply what it measured and remember it. */
function applyCalibration({ calibration, tuning, roundTrip }) {
  head.calibration = calibration;
  Object.assign(head.tuning, tuning);
  if (roundTrip) {
    headSamples.roundTrip = roundTrip;
    $('t-roundtrip').textContent = `${Math.round(roundTrip)} ms`;
  }
  calibratedAt = new Date().toISOString();
  saveTuning();
  syncTuning();
}

function syncTuning() {
  const t = head.tuning;
  $('t-mincutoff').value = t.minCutoff;
  $('t-mincutoff-out').textContent = `${t.minCutoff.toLocaleString('nb-NO', { maximumFractionDigits: 1 })} Hz`;
  $('t-beta').value = t.beta;
  $('t-beta-out').textContent = t.beta.toLocaleString('nb-NO', { maximumFractionDigits: 2 });
  $('t-predict').value = t.predictMs;
  $('t-predict-out').textContent = `${t.predictMs} ms`;
  $('t-delegate').value = t.delegate;
  $('t-depth').value = t.depthCm;
  $('t-depth-out').textContent = `${t.depthCm} cm`;
  $('t-calibrated').textContent = calibratedAt
    ? `ja, ${new Date(calibratedAt).toLocaleString('nb-NO', { dateStyle: 'short', timeStyle: 'short' })}`
    : 'nei';
}

/** "ANGLE (NVIDIA, NVIDIA GeForce RTX 4050 Laptop GPU (0x…) Direct3D11 …)" → "NVIDIA GeForce RTX 4050 Laptop GPU" */
function gpuName() {
  const gl = renderer.gl;
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  const name = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return name.match(/^ANGLE \([^,]+, (.+?) \(0x/)?.[1] ?? name;
}

function bindTuning() {
  loadTuning();
  syncTuning();
  $('t-gpu').textContent = gpuName();
  const bind = (id, key) =>
    $(id).addEventListener('input', (e) => {
      head.tuning[key] = Number(e.target.value);
      saveTuning();
      syncTuning();
    });
  bind('t-mincutoff', 'minCutoff');
  bind('t-beta', 'beta');
  bind('t-predict', 'predictMs');
  bind('t-depth', 'depthCm');
  $('t-delegate').addEventListener('change', async (e) => {
    head.tuning.delegate = e.target.value;
    saveTuning();
    if (!state.head) return;
    head.stop(); // restart the worker with the new delegate
    await setHeadTracking(true);
  });
  $('t-reset').addEventListener('click', () => {
    Object.assign(head.tuning, DEFAULT_TUNING);
    head.calibration = structuredClone(DEFAULT_CALIBRATION);
    calibratedAt = null;
    saveTuning();
    syncTuning();
  });
  $('t-calibrate').addEventListener('click', () => {
    togglePanel(false);
    openCalibration(head, applyCalibration);
  });
  $('t-measure').addEventListener('click', async () => {
    const button = $('t-measure');
    button.disabled = true;
    $('t-roundtrip').textContent = 'måler … se på skjermen';
    try {
      const { median, samples } = await head.measureRoundTrip($('flash'));
      headSamples.roundTrip = median;
      $('t-roundtrip').textContent = `${Math.round(median)} ms (${samples.length} målinger)`;
    } catch (err) {
      $('t-roundtrip').textContent = err.message;
    } finally {
      button.disabled = false;
    }
  });
}

/**
 * Head-coupled perspective as a relief. On a flat screen both eyes see the same image, so they
 * tell the brain everything is on the screen; whatever slides across it reads as the picture
 * moving, not as something far away standing still. (A true window, with real distances, makes
 * distant mountains slide as far as your head moves: on a laptop screen you see almost nothing
 * through it, and every twitch shows.) So the scene is squeezed into a shallow box around the
 * screen: the background (focus) sits on the glass and stays put, nearer things come out
 * towards you and slide against your head movement. The depth comes from how the layers move
 * relative to each other.
 *
 * Physically consistent for a box depthCm deep, seen from your distance: a layer d cm behind the
 * glass slides about d / distance of your head movement.
 */
function headView(rect, dpr) {
  const { pxPerCm, rangeCm } = head.calibration;
  const widthCm = rect.w / dpr / pxPerCm;
  const k = head.tuning.depthCm / (head.distanceCm * widthCm); // painting widths per cm of head movement and unit of depth
  const [hx, hy] = anim.headCm.map((v) => clamp(v, -40, 40));
  return { shift: [hx * k, hy * k], range: rangeCm * k };
}

/** Keep the AI label in the painting's top-right corner, like «Høyoppløst» on nasjonalmuseet.no. */
let badgeStyle = '';
function placeBadge(rect, dpr) {
  const inset = 16;
  const right = Math.min((rect.x + rect.w) / dpr, innerWidth);
  const left = Math.round(right - measured.badgeWidth - inset);
  const top = Math.round(Math.max(rect.y / dpr, measured.area.top) + inset);
  const opacity = (anim.effect * anim.fade).toFixed(2);
  const next = `${left},${top},${opacity}`;
  if (next === badgeStyle) return; // only touch the DOM when something changed
  badgeStyle = next;
  const s = $('ai-badge').style;
  s.left = `${left}px`;
  s.top = `${top}px`;
  s.opacity = opacity;
}

// ---------- Focus by double-click / double-tap ----------

function focusAt(clientX, clientY) {
  if (!current || !lastRect || state.original) return;
  const r = lastRect;
  const ix = (clientX * r.dpr - r.x) / r.w;
  const iy = (clientY * r.dpr - r.y) / r.h;
  if (ix < 0 || ix > 1 || iy < 0 || iy > 1) return;
  const o = renderer.overscan;
  const { w, h, data } = current.depthData;
  const px = clamp(Math.floor((0.5 + (ix - 0.5) * (1 - 2 * o)) * w), 0, w - 1);
  const py = clamp(Math.floor((0.5 + (iy - 0.5) * (1 - 2 * o)) * h), 0, h - 1);
  state.focusTarget = data[(py * w + px) * 4] / 255;
  anim.focusFlash = 1.6;
  syncPanel();
  toast(`Fokusplan ${state.focusTarget.toFixed(2)}`);
}

// ---------- Clip recording ----------

function pickMime() {
  const options = ['video/mp4;codecs=avc1.640028', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  return options.find((m) => MediaRecorder.isTypeSupported(m));
}

async function recordClip() {
  if (state.recording) return;
  if (!('MediaRecorder' in window) || !pickMime()) {
    toast('Nettleseren kan ikke ta opp video');
    return;
  }
  const work = state.works[state.index];
  const size = CLIP_FORMATS[state.clipFormat];
  const mime = pickMime();
  state.original = false;

  // Composite the WebGL frame with logo and caption on a 2D canvas, and record that
  const out = document.createElement('canvas');
  [out.width, out.height] = size;
  const ctx = out.getContext('2d');
  const compose = (w) => {
    const u = Math.min(out.width, out.height) / 1080;
    const pad = Math.round(56 * u);
    ctx.drawImage(canvas, 0, 0);
    const g = ctx.createLinearGradient(0, out.height * 0.62, 0, out.height);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(0,0,0,0.72)');
    ctx.fillStyle = g;
    ctx.fillRect(0, out.height * 0.62, out.width, out.height * 0.38);
    if (logo.complete) ctx.drawImage(logo, pad, pad, 52 * u * (logo.width / logo.height || 0.8), 52 * u);
    ctx.fillStyle = '#fff';
    ctx.font = `500 ${Math.round(52 * u)}px MuseetSans, Helvetica, sans-serif`;
    ctx.fillText(w.title, pad, out.height - pad - 84 * u);
    ctx.font = `400 ${Math.round(30 * u)}px MuseetSans, Helvetica, sans-serif`;
    ctx.fillText([w.artist, w.date].filter(Boolean).join(', '), pad, out.height - pad - 38 * u);
    ctx.fillStyle = '#afafaf';
    ctx.font = `400 ${Math.round(22 * u)}px MuseetSans, Helvetica, sans-serif`;
    ctx.fillText(`${w.institution ?? 'Nasjonalmuseet'} · Dybdeeffekt laget med KI`, pad, out.height - pad);
  };

  const stream = out.captureStream(60);
  const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16_000_000 });
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise((r) => (recorder.onstop = r));

  state.recording = { size, start: performance.now(), ms: CLIP_SECONDS * 1000, compose };
  document.body.classList.add('recording');
  const button = $('record');
  button.disabled = true;
  button.firstElementChild.textContent = 'Tar opp …';
  recorder.start();
  await new Promise((r) => setTimeout(r, CLIP_SECONDS * 1000));
  recorder.stop();
  await stopped;
  stream.getTracks().forEach((track) => track.stop());
  state.recording = null;
  document.body.classList.remove('recording');
  button.disabled = false;
  button.firstElementChild.textContent = 'Lag klipp';
  syncPanel();

  const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(chunks, { type: mime }));
  a.download = `${work.slug}-levende-${state.clipFormat.replace(':', 'x')}.${ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  toast(`Lagret ${a.download}`);
}

// ---------- UI ----------

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

/** "Nasjonalmuseet/Høstland, Børre" → "Børre Høstland" */
function photographer(credit) {
  const name = (credit || '').replace(/^Nasjonalmuseet\/?/, '').trim();
  const m = name.match(/^([^,]+),\s*(.+)$/);
  return m ? `${m[2]} ${m[1]}` : name;
}

function creditLine(work) {
  if (work.institution === 'Munchmuseet') return 'Foto © Munchmuseet';
  const by = photographer(work.photoCredit);
  // Photographs carry the collection name instead of a reproduction photographer
  if (by.startsWith('Nasjonalmuseet')) return `${by} (${work.license})`;
  return `Foto: ${by ? `${by}, ` : ''}Nasjonalmuseet (${work.license})`;
}

function renderCaption(work) {
  $('category').textContent = work.genre || 'Maleri';
  $('title').textContent = work.title;
  $('title').lang = 'nb';

  const creators = work.creators?.length ? work.creators : [{ name: work.artist, url: '' }];
  $('artists').replaceChildren(
    ...creators.map((c) =>
      c.url ? el('a', { href: c.url, target: '_blank', rel: 'noopener', textContent: c.name }) : el('span', { textContent: c.name }),
    ),
  );
  $('date').textContent = work.date;

  $('credit').replaceChildren(
    `${creditLine(work)} · `,
    el('a', { href: work.pageUrl, target: '_blank', rel: 'noopener' }, [
      work.institution === 'Munchmuseet' ? 'Les mer hos MUNCH' : 'Se verket i samlingen',
      el('span', { className: 'icon icon--external-link', ariaHidden: 'true' }),
    ]),
  );

  $('model').textContent = work.depthModel ?? '–';
  $('counter').textContent = `${state.index + 1} / ${state.works.length}`;
  const strip = $('thumbs');
  strip.querySelectorAll('button').forEach((b, i) => {
    b.setAttribute('aria-current', String(i === state.index));
    if (i === state.index) {
      strip.scrollTo({ left: b.parentElement.offsetLeft - (strip.clientWidth - b.offsetWidth) / 2, behavior: 'smooth' });
    }
  });
}

function syncPanel() {
  document.querySelector(`input[name="mode"][value="${state.mode}"]`).checked = true;
  $('mode-hint').textContent = MODE_HINTS[state.mode];
  $('strength').value = (state.strength * 100).toFixed(1);
  $('strength-out').textContent = `${(state.strength * 100).toFixed(1)} %`;
  $('focus').value = state.focusTarget.toFixed(2);
  $('focus-out').textContent = state.focusTarget.toFixed(2);
  $('steps').value = state.maxSteps;
  $('steps-out').textContent = state.maxSteps;
  $('breathe').checked = state.breathe;
  $('dust').checked = state.dust;
  $('dolly').checked = state.dolly;
  $('depth').checked = state.showDepth;
  $('fill').checked = state.fit === 'fill';
  document.body.classList.toggle('fill', state.fit === 'fill');
  $('head').checked = state.head;
  $('clip-format').value = state.clipFormat;
  const orig = $('original-btn');
  orig.setAttribute('aria-pressed', String(state.original));
  orig.lastElementChild.textContent = state.original ? 'Vis levende' : 'Vis original';
}

/**
 * The depth that stays still. Following the head, it's the background: the
 * bulk of the picture stays calm on the screen and what's nearer comes towards you.
 */
function defaultFocus(work) {
  if (state.head && current) return work.headFocus ?? current.background;
  return work.focus ?? 0.5;
}

function resetFocus() {
  const work = state.works[state.index];
  if (!work) return;
  state.focusTarget = defaultFocus(work);
  syncPanel();
}

async function setHeadTracking(on) {
  state.head = on;
  $('head-preview').hidden = !on;
  resetFocus();
  if (!on) {
    head.stop();
    $('head-tuning').hidden = true;
    return;
  }
  toast('Starter kameraet …');
  try {
    await head.start($('head-video'), $('head-dot'));
    $('head-tuning').hidden = false;
    if (!calibratedAt && !KIOSK) {
      // First time: calibrate for the best effect
      togglePanel(false);
      openCalibration(head, applyCalibration);
    }
    else toast('Beveg hodet for å se rundt i bildet');
  } catch (err) {
    console.error(err);
    state.head = false;
    $('head-preview').hidden = true;
    $('head-tuning').hidden = true;
    resetFocus();
    const messages = { NotAllowedError: 'Tilgang til kameraet ble avslått', NotFoundError: 'Fant ikke noe kamera' };
    toast(messages[err.name] ?? (err.message || 'Kunne ikke starte kameraet'));
  }
}

let toastTimer;
function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
}

function togglePanel(open = $('panel').hidden) {
  $('panel').hidden = !open;
  $('panel-btn').setAttribute('aria-expanded', String(open));
  if (open && !$('library').hidden) toggleLibrary(false);
}

function fail(message, err) {
  console.error(err);
  const box = $('error');
  box.hidden = false;
  box.replaceChildren(
    el('div', {}, [
      el('h2', { textContent: message }),
      el('p', { innerHTML: 'Kjør <code>tools/fetch_artworks.py</code> og <code>tools/make_depth.py</code> først. Se README.' }),
    ]),
  );
}

function renderThumbs() {
  $('thumbs').replaceChildren(
    ...state.works.map((w, i) => {
      const b = el('button', { title: `${w.title} – ${w.artist}` });
      b.setAttribute('aria-label', b.title);
      b.setAttribute('aria-current', String(i === state.index));
      b.style.backgroundImage = `url("${w.image}")`;
      b.addEventListener('click', () => show(i));
      return el('li', {}, [b]);
    }),
  );
}

// ---------- «Hent fra samlingen»: search the collection, depth map made by tools/server.py ----------

function toggleLibrary(open = $('library').hidden) {
  $('library').hidden = !open;
  $('library-btn').setAttribute('aria-expanded', String(open));
  if (open) {
    togglePanel(false);
    $('search-q').focus();
  }
}

async function searchCollection(query, type) {
  const status = $('search-status');
  status.textContent = 'Søker …';
  $('results').replaceChildren();
  try {
    const res = await fetch(`api/search?${new URLSearchParams({ q: query, type })}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    status.textContent = data.items.length
      ? `${data.items.length} treff. Velg et verk for å gi det dybde.`
      : 'Ingen treff med bilde.';
    $('results').replaceChildren(...data.items.map(resultCard));
  } catch (err) {
    status.textContent = err.message || 'Søket feilet.';
  }
}

function resultCard(item) {
  const card = el('button', { className: 'result', type: 'button' }, [
    el('img', { src: item.thumb, alt: '', loading: 'lazy' }),
    el('span', { className: 'title', textContent: item.title }),
    el('span', { className: 'meta', textContent: [item.artist, item.date].filter(Boolean).join(', ') }),
  ]);
  if (item.added) card.append(el('span', { className: 'badge', textContent: 'I demoen' }));
  card.addEventListener('click', () => addFromCollection(item, card));
  return el('li', {}, [card]);
}

async function addFromCollection(item, card) {
  const known = state.works.findIndex((w) => w.id === item.id);
  if (known >= 0) {
    toggleLibrary(false);
    show(known);
    return;
  }
  if (card.classList.contains('busy')) return;
  card.classList.add('busy');
  const status = el('span', { className: 'status', textContent: 'Starter …' });
  card.append(status);

  try {
    const res = await fetch('api/works', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: item.id }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);

    // NDJSON: one progress message per line, ending with {"done": work} or {"error": …}
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    let work = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let nl;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const msg = JSON.parse(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
        if (msg.error) throw new Error(msg.error);
        if (msg.message) status.textContent = msg.message;
        if (msg.done) work = msg.done;
      }
    }
    if (!work) throw new Error('Serveren avsluttet uten resultat');

    state.works.push(work);
    renderThumbs();
    toggleLibrary(false);
    await show(state.works.length - 1);
    toast(`${work.title} er lagt til`);
    card.classList.remove('busy');
    status.remove();
    card.append(el('span', { className: 'badge', textContent: 'I demoen' }));
  } catch (err) {
    card.classList.remove('busy');
    status.classList.add('error');
    status.textContent = err.message;
  }
}

async function bindLibrary() {
  try {
    const res = await fetch('api/health');
    if (!res.ok || !(await res.json()).ok) return;
  } catch {
    return; // static hosting: no depth server, no button
  }
  $('library-btn').hidden = false;
  $('library-btn').addEventListener('click', () => toggleLibrary());
  $('library-close').addEventListener('click', () => toggleLibrary(false));
  $('search-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('search-q').value.trim();
    if (q) searchCollection(q, $('search-type').value);
  });
}

function bindUI() {
  const toggles = { breathe: 'breathe', dust: 'dust', dolly: 'dolly', depth: 'showDepth' };
  for (const [id, key] of Object.entries(toggles)) {
    $(id).addEventListener('change', (e) => (state[key] = e.target.checked));
  }
  $('fill').addEventListener('change', (e) => {
    state.fit = e.target.checked ? 'fill' : 'contain';
    syncPanel();
  });
  $('head').addEventListener('change', (e) => setHeadTracking(e.target.checked));
  bindTuning();
  document.querySelectorAll('input[name="mode"]').forEach((r) =>
    r.addEventListener('change', (e) => {
      state.mode = Number(e.target.value);
      syncPanel();
    }),
  );
  $('strength').addEventListener('input', (e) => {
    state.strength = Number(e.target.value) / 100;
    syncPanel();
  });
  $('focus').addEventListener('input', (e) => {
    state.focusTarget = Number(e.target.value);
    anim.focusFlash = 1;
    syncPanel();
  });
  $('steps').addEventListener('input', (e) => {
    state.maxSteps = Number(e.target.value);
    syncPanel();
  });
  $('clip-format').addEventListener('change', (e) => (state.clipFormat = e.target.value));
  $('record').addEventListener('click', recordClip);
  $('panel-btn').addEventListener('click', () => togglePanel());
  $('panel-close').addEventListener('click', () => togglePanel(false));
  $('prev').addEventListener('click', () => step(-1));
  $('next').addEventListener('click', () => step(1));
  $('original-btn').addEventListener('click', () => {
    state.original = !state.original;
    syncPanel();
  });

  renderThumbs();
  bindLibrary();

  canvas.addEventListener('dblclick', (e) => focusAt(e.clientX, e.clientY));
  let lastTap = null;
  canvas.addEventListener('pointerup', (e) => {
    if (e.pointerType === 'mouse') return;
    const now = performance.now();
    if (lastTap && now - lastTap.t < 320 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
      focusAt(e.clientX, e.clientY);
      lastTap = null;
    } else {
      lastTap = { t: now, x: e.clientX, y: e.clientY };
    }
  });

  const gyroBtn = $('gyro-btn');
  if (input.gyroAvailable && matchMedia('(pointer: coarse)').matches) {
    gyroBtn.hidden = false;
    gyroBtn.addEventListener('click', async () => {
      try {
        await input.enableGyro();
        gyroBtn.hidden = true;
        toast('Vipp telefonen for å se rundt i maleriet');
      } catch (err) {
        toast(err.message);
      }
    });
  }

  // In "fill" the UI covers the painting, so it hides when the mouse rests (like a video player)
  let idleTimer;
  const wakeUI = () => {
    document.body.classList.remove('idle');
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      if ($('panel').hidden && $('library').hidden) document.body.classList.add('idle');
    }, 2500);
  };
  for (const type of ['pointermove', 'pointerdown', 'keydown']) window.addEventListener(type, wakeUI, { passive: true });
  wakeUI();

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      togglePanel(false);
      if (!$('library').hidden) toggleLibrary(false);
      if (!$('calibration').hidden) closeCalibration();
      return;
    }
    if (!$('calibration').hidden) return; // keys belong to the calibration while it runs
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // Arrows change the work everywhere except where they edit something (text, sliders, lists)
    const editing = e.target.closest('input:not([type="checkbox"]):not([type="radio"]), select, textarea');
    if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && !editing) {
      e.preventDefault();
      step(e.key === 'ArrowRight' ? 1 : -1);
      return;
    }
    if (e.target.closest('input, select, textarea')) return;
    const key = e.key.toLowerCase();
    const flip = (k, label) => {
      state[k] = !state[k];
      toast(`${label} ${state[k] ? 'på' : 'av'}`);
    };
    if (key === 'o') flip('original', 'Original');
    else if (key === 'm') {
      state.mode = state.mode ? 0 : 1;
      toast(state.mode ? 'Okklusjon (POM)' : 'Enkel forskyvning');
    } else if (key === 'd') flip('showDepth', 'Dybdekart');
    else if (key === 'b') flip('breathe', 'Pust');
    else if (key === 's') flip('dust', 'Støv');
    else if (key === 'z') flip('dolly', 'Dolly zoom');
    else if (key === 'f') state.fit = state.fit === 'fill' ? 'contain' : 'fill';
    else if (key === 'c') setHeadTracking(!state.head);
    else if (key === 'h') document.body.classList.toggle('ui-hidden');
    else if (key === 'r') recordClip();
    else return;
    syncPanel();
  });
}

// ---------- Start ----------

async function main() {
  try {
    renderer = new ParallaxRenderer(canvas);
  } catch (e) {
    fail(e.message, e);
    return;
  }
  renderer.background = [0, 0, 0];
  input = new TiltInput(canvas);
  if (EMBED || KIOSK) document.body.classList.add('ui-hidden');
  measureLayout();
  const observer = new ResizeObserver(measureLayout);
  observer.observe($('frame-area'));
  observer.observe($('ai-badge'));
  window.addEventListener('resize', measureLayout);
  document.fonts.ready.then(measureLayout);

  let manifest;
  try {
    const res = await fetch('art/works.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(res.status);
    manifest = await res.json();
  } catch (e) {
    fail('Fant ingen verk (art/works.json)', e);
    return;
  }
  state.works = manifest.works;
  if (!state.works.length) return fail('art/works.json er tom', null);

  bindUI();
  const wanted = params.get('work')?.toLowerCase();
  const start = Math.max(0, state.works.findIndex((w) => w.slug === wanted || w.id.toLowerCase() === wanted));
  await show(start);
  requestAnimationFrame(frame);

  if (params.has('head')) setHeadTracking(true); // e.g. ?kiosk&head on a museum screen with a camera
}

main();
