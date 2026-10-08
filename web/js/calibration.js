// Calibration for head tracking. The layers should move by a physically consistent amount, so
// this measures the screen and your head in centimetres:
//   1. Card on the screen → screen size (CSS pixels per centimetre)
//   2. Sit still and look at the centre → where "straight on" is, your pupil distance in the
//      image (the ruler for centimetres), and how much the tracking jitters (→ minCutoff)
//   3. Sway from side to side → how far and how fast you move (→ cropping, beta)
//   4. Screen flashes → screen + camera latency (→ how far ahead to predict)

import { IPD_CM, PHONE, distanceFromPupils } from './headtrack.js';

const $ = (id) => document.getElementById(id);

const CARD_CM = [8.56, 5.4];  // ID-1 card: bank cards, ID cards
const STILL_MS = 3000;
const SWAY_MS = 6000;
// Head jitter left after filtering. The nearest layer moves about depth/distance (~1/6) of your
// head movement, so this is well under a pixel on screen; less smoothing also means less lag.
const NOISE_TARGET_CM = 0.08;
const MOVING_CUTOFF_HZ = 8;   // the filter should open up to this when you move at your usual speed
const PREDICT_SHARE = 0.8;    // predict a bit less than the measured latency: overshooting looks worse than lag
const MAX_PX_PER_CM = 80;     // top of the screen-size slider

let run = 0;

/**
 * @param {import('./headtrack.js').HeadTracker} head
 * @param {(result: object) => void} onSaved  called with the new calibration and tuning
 */
export function openCalibration(head, onSaved) {
  $('calibration').hidden = false;
  show({
    step: 'Kalibrering · cirka 30 sekunder',
    title: 'Kalibrer hodesporingen',
    text:
      'For at dybden skal se riktig ut, må vi vite hvor stor skjermen er og hvor du er. Ha et bankkort klart' +
      (PHONE
        ? ', hold telefonen 30–40 cm fra ansiktet og ha lys på ansiktet.'
        : ', sitt 50–70 cm fra skjermen og ha lys på ansiktet.'),
    primary: ['Start', () => calibrate(head, onSaved)],
  });
}

export function closeCalibration() {
  run++; // stops a calibration in progress
  $('calibration').hidden = true;
  $('cal-target').hidden = true;
  $('cal-ruler').hidden = true;
}

function show({ step = '', title, text, primary = null, progressMs = 0, target = false, ruler = false, card = true, cancel = true }) {
  $('cal-step').textContent = step;
  $('cal-title').textContent = title;
  $('cal-text').textContent = text;
  $('cal-card').hidden = !card;
  $('cal-target').hidden = !target;
  $('cal-ruler').hidden = !ruler;
  $('cal-cancel').hidden = !cancel;
  const button = $('cal-primary');
  button.hidden = !primary;
  if (primary) {
    button.firstElementChild.textContent = primary[0];
    button.onclick = primary[1];
  }
  $('cal-cancel').onclick = closeCalibration;
  const bar = $('cal-bar');
  bar.style.transition = 'none';
  bar.style.width = '0%';
  bar.parentElement.hidden = !progressMs;
  if (progressMs) {
    bar.getBoundingClientRect(); // restart the transition
    bar.style.transition = `width ${progressMs}ms linear`;
    bar.style.width = '100%';
  }
}

const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
const std = (xs) => {
  const m = xs.reduce((s, x) => s + x, 0) / xs.length;
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
};
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const nb = (v, digits = 1) => v.toLocaleString('nb-NO', { maximumFractionDigits: digits });

/**
 * Step 1: resize a card outline until it matches a real card held against the screen. On a
 * screen narrower than a card (a phone held upright) the card stands on its short side.
 */
function measureScreen(pxPerCm) {
  return new Promise((resolve) => {
    const slider = $('cal-scale');
    const outline = $('cal-card-outline');
    const upright = innerWidth < CARD_CM[0] * MAX_PX_PER_CM + 32;
    const [w, h] = upright ? [CARD_CM[1], CARD_CM[0]] : CARD_CM;
    outline.firstElementChild.textContent = `${nb(w * 10)} mm`;
    const draw = () => {
      const scale = Number(slider.value);
      outline.style.width = `${w * scale}px`;
      outline.style.height = `${h * scale}px`;
    };
    slider.max = MAX_PX_PER_CM;
    slider.value = pxPerCm;
    slider.oninput = draw;
    draw();
    show({
      step: 'Steg 1 av 4',
      title: upright ? 'Hold et bankkort på høykant mot skjermen' : 'Hold et bankkort mot skjermen',
      text: 'Dra glidebryteren til rammen er nøyaktig like bred som kortet. Da vet vi hvor stor skjermen er.',
      ruler: true,
      primary: ['Neste', () => resolve(Number(slider.value))],
    });
  });
}

async function calibrate(head, onSaved) {
  const id = ++run;
  const alive = () => id === run;
  try {
    const pxPerCm = await measureScreen(head.calibration.pxPerCm);
    if (!alive()) return;

    // 2. Still
    show({
      step: 'Steg 2 av 4',
      title: 'Se på prikken og hold hodet i ro',
      text: 'Slik finner vi hvor «rett forfra» er, hvor stort ansiktet ditt er i kamerabildet, og hvor mye sporingen skjelver.',
      progressMs: STILL_MS,
      target: true,
    });
    const still = await head.collect(STILL_MS);
    if (!alive()) return;
    if (still.length < 10) throw new Error('Fant ikke ansiktet ditt. Sjekk at du er midt i kamerabildet og har lys på ansiktet.');
    const center = [median(still.map((s) => s.cx)), median(still.map((s) => s.cy))];
    const pupils = median(still.map((s) => s.pupils));
    const cmPerWidth = IPD_CM / pupils;
    const aspect = head.aspect;
    const toCm = (s) => [(center[0] - s.cx) * cmPerWidth, (s.cy - center[1]) * aspect * cmPerWidth];
    const stillCm = still.map(toCm);
    const noise = (std(stillCm.map((p) => p[0])) + std(stillCm.map((p) => p[1]))) / 2;
    const rate = still.length / (STILL_MS / 1000);
    // A first-order low-pass at fc keeps about π·fc/rate of white noise's variance
    const minCutoff = clamp((rate / Math.PI) * (NOISE_TARGET_CM / Math.max(noise, 1e-4)) ** 2, 0.2, 4);
    const distanceCm = distanceFromPupils(pupils, aspect);

    // 3. Sway
    show({
      step: 'Steg 3 av 4',
      title: PHONE ? 'Beveg hodet eller telefonen rolig fra side til side' : 'Beveg hodet rolig fra side til side',
      text: 'Så langt og så fort det føles naturlig når du ser på et bilde. Litt opp og ned også.',
      progressMs: SWAY_MS,
    });
    const sway = await head.collect(SWAY_MS);
    if (!alive()) return;
    if (sway.length < 20) throw new Error('Mistet ansiktet ditt underveis. Prøv å holde deg innenfor kamerabildet.');
    const swayCm = sway.map((s) => ({ p: toCm(s), t: s.t }));
    const reach = Math.max(...swayCm.map(({ p }) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
    const rangeCm = clamp(reach * 1.15, 2, 30);
    const speeds = [];
    for (let i = 2; i < swayCm.length; i++) {
      const a = swayCm[i - 2]; // two samples apart: less noise in the speed
      const b = swayCm[i];
      speeds.push(Math.hypot(b.p[0] - a.p[0], b.p[1] - a.p[1]) / Math.max(0.001, (b.t - a.t) / 1000));
    }
    const speed = Math.max(2, speeds.sort((a, b) => a - b)[Math.floor(speeds.length * 0.7)]);
    const beta = clamp((MOVING_CUTOFF_HZ - minCutoff) / speed, 0.02, 3);

    // 4. Latency
    show({
      step: 'Steg 4 av 4',
      title: 'Se på skjermen – den blinker',
      text: 'Vi måler hvor lang tid det tar fra skjermen lyser til kameraet ser det.',
    });
    await wait(1500);
    if (!alive()) return;
    show({ title: '', text: '', card: false });
    let roundTrip = null;
    let predictMs = head.tuning.predictMs;
    try {
      roundTrip = (await head.measureRoundTrip($('flash'))).median;
      // The time between capture and the browser getting the frame is already extrapolated
      predictMs = Math.round(clamp(PREDICT_SHARE * (roundTrip - head.stats.captureLagMs), 0, 150) / 5) * 5;
    } catch {
      // too bright to see the flashes: keep the current prediction
    }
    if (!alive()) return;

    const result = {
      calibration: { center, pupils, rangeCm, pxPerCm },
      tuning: { minCutoff, beta, predictMs },
      roundTrip,
    };
    onSaved(result);
    show({
      step: 'Ferdig',
      title: 'Kalibrert',
      text:
        `Du er cirka ${Math.round(distanceCm)} cm unna og beveger deg ±${Math.round(reach)} cm; ` +
        'bildet beskjæres akkurat nok til at kantene ikke vises. ' +
        `Ro ${nb(minCutoff)} Hz, respons ${nb(beta, 2)}, prediksjon ${predictMs} ms` +
        (roundTrip ? ` (skjerm → kamera ${Math.round(roundTrip)} ms).` : ' (blinkene ble ikke sett – prøv i et mørkere rom).'),
      primary: ['Ferdig', closeCalibration],
      cancel: false,
    });
  } catch (err) {
    if (!alive()) return;
    show({ step: 'Kalibreringen stoppet', title: 'Noe gikk galt', text: err.message, primary: ['Prøv igjen', () => calibrate(head, onSaved)] });
  }
}
