// Turns mouse, touch drag and device orientation into a camera "tilt" in roughly -1..1.
// +x means the viewer's head moves right, +y means it moves down.

const GYRO_RANGE_DEG = 18;     // tilt in degrees that maps to full parallax
const GYRO_RECENTER_S = 4;     // the neutral pose slowly follows how you hold the phone
const DEG = Math.PI / 180;
const clamp = (v, lo = -1, hi = 1) => Math.min(hi, Math.max(lo, v));

// Quaternions [w, x, y, z]. The tilt is the rotation since the neutral pose, taken from the full
// orientation instead of the beta/gamma angles alone: those are Euler angles, and with the phone
// held upright (beta ≈ 90°) side-to-side turns land in alpha or flip gamma's sign.
const multiply = ([aw, ax, ay, az], [bw, bx, by, bz]) => [
  aw * bw - ax * bx - ay * by - az * bz,
  aw * bx + ax * bw + ay * bz - az * by,
  aw * by - ax * bz + ay * bw + az * bx,
  aw * bz + ax * by - ay * bx + az * bw,
];
const conjugate = ([w, x, y, z]) => [w, -x, -y, -z];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
const normalize = (q) => {
  const n = Math.hypot(...q);
  return q.map((v) => v / n);
};

/** deviceorientation angles (Z-X'-Y'', degrees) → quaternion from device to world. */
function orientation(alpha, beta, gamma) {
  const half = (deg) => (deg * DEG) / 2;
  const z = [Math.cos(half(alpha)), 0, 0, Math.sin(half(alpha))];
  const x = [Math.cos(half(beta)), Math.sin(half(beta)), 0, 0];
  const y = [Math.cos(half(gamma)), 0, Math.sin(half(gamma)), 0];
  return multiply(multiply(z, x), y);
}

/** Rotation from `from` to `to`, about the device's own axes: [about x, about y] in degrees. */
function tiltBetween(from, to) {
  let [w, x, y, z] = multiply(conjugate(from), to);
  if (w < 0) [w, x, y, z] = [-w, -x, -y, -z];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0];
  const k = (2 * Math.atan2(s, w)) / s / DEG; // rotation vector, in degrees
  return [x * k, y * k];
}

export class TiltInput {
  target = [0, 0];
  lastActivity = 0;
  gyroActive = false;

  #baseline = null;
  #lastGyroTime = 0;
  #drag = null;

  constructor(surface) {
    this.surface = surface;

    window.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'mouse') this.#fromMouse(e);
      else if (this.#drag && e.pointerId === this.#drag.id) this.#fromDrag(e);
    });
    document.documentElement.addEventListener('pointerleave', (e) => {
      if (e.pointerType === 'mouse') this.target = [0, 0];
    });
    surface.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse' || this.gyroActive) return;
      this.#drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
      surface.setPointerCapture(e.pointerId);
    });
    const endDrag = (e) => {
      if (this.#drag && e.pointerId === this.#drag.id) {
        this.#drag = null;
        this.target = [0, 0];
      }
    };
    surface.addEventListener('pointerup', endDrag);
    surface.addEventListener('pointercancel', endDrag);
  }

  get gyroAvailable() {
    return 'DeviceOrientationEvent' in window;
  }

  /** Must be called from a user gesture (tap) – iOS asks for permission here. */
  async enableGyro() {
    if (!window.isSecureContext) throw new Error('Bevegelsessensoren krever HTTPS (bruk «npm run dev»).');
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      const answer = await DeviceOrientationEvent.requestPermission();
      if (answer !== 'granted') throw new Error('Tilgang til bevegelsessensoren ble avslått.');
    }
    window.addEventListener('deviceorientation', (e) => this.#fromOrientation(e));
    this.gyroActive = true;
    this.#baseline = null;
  }

  #touch() {
    this.lastActivity = performance.now();
  }

  #fromMouse(e) {
    this.target = [clamp((e.clientX / innerWidth) * 2 - 1), clamp((e.clientY / innerHeight) * 2 - 1)];
    this.#touch();
  }

  #fromDrag(e) {
    const s = Math.min(innerWidth, innerHeight) * 0.35;
    this.target = [clamp((e.clientX - this.#drag.x) / s), clamp((e.clientY - this.#drag.y) / s)];
    this.#touch();
  }

  #fromOrientation(e) {
    if (e.beta == null || e.gamma == null) return;
    const q = orientation(e.alpha ?? 0, e.beta, e.gamma);

    // The neutral pose slowly follows how you hold the phone
    const now = performance.now();
    const dt = Math.min(0.2, (now - (this.#lastGyroTime || now)) / 1000);
    this.#lastGyroTime = now;
    if (!this.#baseline) this.#baseline = q;
    const k = 1 - Math.exp(-dt / GYRO_RECENTER_S);
    const sign = dot(this.#baseline, q) < 0 ? -1 : 1; // q and -q are the same rotation
    this.#baseline = normalize(this.#baseline.map((v, i) => v + (sign * q[i] - v) * k));

    // About the device's x axis: top edge towards you. About its y axis: right edge away.
    // Mapped to the screen's axes so it works in landscape too.
    const [ax, ay] = tiltBetween(this.#baseline, q);
    const angle = ((screen.orientation?.angle ?? window.orientation ?? 0) + 360) % 360;
    let [x, y] = [ay, ax];
    if (angle === 90) [x, y] = [ax, -ay];
    else if (angle === 180) [x, y] = [-ay, -ax];
    else if (angle === 270) [x, y] = [-ax, ay];

    // Like a viewfinder: turn the phone towards something to see more of it. Right edge away
    // (the back points left) shows more of the left; top edge towards you shows more of the top.
    this.target = [clamp(x / GYRO_RANGE_DEG), clamp(y / GYRO_RANGE_DEG)];
    this.#touch();
  }
}
