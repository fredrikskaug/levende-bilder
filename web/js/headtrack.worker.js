// Face tracking off the main thread, with MediaPipe Face Landmarker. In video mode it tracks the
// face from the previous frame instead of detecting it anew each time, which is what keeps the
// position from jumping. A detection can take tens of milliseconds; here it can't stall rendering.
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

let candidates = [];      // [{ delegate, landmarker, times, calls }]; one left once chosen
let turn = 0;
let frames = 0;

function create(FaceLandmarker, fileset, model, delegate) {
  // The wasm loader hands over its module once per import; a URL of its own per delegate makes
  // it run again, otherwise the second landmarker fails with "ModuleFactory not set"
  const own = { ...fileset, wasmLoaderPath: `${fileset.wasmLoaderPath}?${delegate}` };
  return FaceLandmarker.createFromOptions(own, {
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];

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
  self.postMessage({ type: 'delegate', delegate: best.delegate, ms: timings });
}

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init') {
    try {
      const { FilesetResolver, FaceLandmarker } = await import(`${msg.base}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${msg.base}/wasm`, true); // ES module build for module workers
      const wanted = msg.delegate === 'auto' ? ['GPU', 'CPU'] : [msg.delegate];
      for (const delegate of wanted) {
        try {
          candidates.push({ delegate, landmarker: await create(FaceLandmarker, fileset, msg.model, delegate), times: [], calls: 0 });
        } catch {
          // no WebGL2 in workers here, for example
        }
      }
      if (!candidates.length) candidates.push({ delegate: 'CPU', landmarker: await create(FaceLandmarker, fileset, msg.model, 'CPU'), times: [], calls: 0 });
      const delegate = candidates.length > 1 ? 'auto' : candidates[0].delegate;
      self.postMessage({ type: 'ready', delegate });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err?.message ?? err) });
    }
    return;
  }

  if (msg.type === 'frame') {
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
    self.postMessage({ type: 'result', t: msg.t, ms, anchor, pupils });
    return;
  }

  if (msg.type === 'close') {
    candidates.forEach((c) => c.landmarker.close());
    self.close();
  }
};
