// GLSL for the parallax painting.
//
// Coordinate spaces (all with origin top-left, y down):
//   img  – 0..1 across the painting as it's laid out on screen
//   q    – img zoomed in slightly (overscan) so displaced lookups never leave the texture
//   uv   – where we actually sample the texture after parallax
//
// A point on the painting at texture position uv with depth h (0 = far, 1 = near)
// is seen on screen where project(q, h) == uv. Moving the camera (uShift) moves
// points proportionally to (h - uFocus), so the focus plane stays still.

const COMMON = /* glsl */ `
uniform vec2 uOverscan;   // fraction cropped from each side
uniform vec2 uShift;      // camera offset, in uv per unit of depth
uniform float uFocus;     // depth that doesn't move
uniform float uDolly;     // dolly zoom: >0 enlarges the foreground and shrinks the background

vec2 imgToQ(vec2 img) { return 0.5 + (img - 0.5) * (1.0 - 2.0 * uOverscan); }
vec2 qToImg(vec2 q) { return 0.5 + (q - 0.5) / (1.0 - 2.0 * uOverscan); }

// Where in the texture does the view ray through q hit the layer at depth z?
vec2 project(vec2 q, float z) {
  float dz = z - uFocus;
  return 0.5 + (q - 0.5) * (1.0 - uDolly * dz) + uShift * dz;
}

// Inverse of project(): where on screen (q) does texture point uv at depth z end up?
vec2 unproject(vec2 uv, float z) {
  float dz = z - uFocus;
  return 0.5 + (uv - 0.5 - uShift * dz) / (1.0 - uDolly * dz);
}
`;

export const fullscreenVert = /* glsl */ `#version 300 es
// One oversized triangle covers the viewport, no buffers needed
out vec2 vImg;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vImg = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const paintingFrag = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D uImage;
uniform sampler2D uDepth;
uniform int uMode;          // 0 = naive offset, 1 = parallax occlusion mapping
uniform int uSteps;         // ray-march steps for mode 1
uniform float uShowDepth;   // 0..1 blend towards the depth visualisation
uniform float uFocusFlash;  // 0..1 highlights the focus plane
uniform float uFade;        // 0..1 used when switching paintings
uniform vec3 uBackground;

${COMMON}

in vec2 vImg;
out vec4 outColor;

float depthAt(vec2 uv) { return textureLod(uDepth, uv, 0.0).r; }

// Naive version from the pitch: uv += (depth - focus) * tilt * strength.
// It reads depth where the pixel *is*, not where it comes from, so edges smear.
vec2 parallaxNaive(vec2 q, out float hit) {
  hit = depthAt(q);
  return project(q, hit);
}

// Parallax occlusion mapping: march the view ray from the nearest layer (z = 1)
// backwards and stop at the first surface it hits. Foreground correctly covers
// background, and disoccluded areas get stretched from the background side.
vec2 parallaxOcclusion(vec2 q, out float hit) {
  float stepSize = 1.0 / float(uSteps);
  float z = 1.0;
  float h = depthAt(project(q, z));
  float prevZ = z;
  float prevH = h;
  for (int i = 0; i < 256; i++) {
    if (h >= z || i >= uSteps) break;
    prevZ = z;
    prevH = h;
    z -= stepSize;
    h = depthAt(project(q, z));
  }
  // Linear refinement between the last step in front of the surface and the first behind it
  float before = prevZ - prevH;
  float after = z - h;
  float t = clamp(before / max(before - after, 1e-5), 0.0, 1.0);
  hit = mix(prevZ, z, t);
  return project(q, hit);
}

// Depth palette: far = deep blue, near = warm white
vec3 depthPalette(float t) {
  vec3 a = vec3(0.05, 0.06, 0.20);
  vec3 b = vec3(0.32, 0.15, 0.52);
  vec3 c = vec3(0.93, 0.42, 0.28);
  vec3 d = vec3(1.00, 0.93, 0.70);
  vec3 col = mix(a, b, smoothstep(0.0, 0.35, t));
  col = mix(col, c, smoothstep(0.3, 0.7, t));
  return mix(col, d, smoothstep(0.65, 1.0, t));
}

void main() {
  vec2 q = imgToQ(vImg);
  float hit;
  vec2 uv = uMode == 0 ? parallaxNaive(q, hit) : parallaxOcclusion(q, hit);

  // Use the undisplaced gradients for mip selection, otherwise depth edges pick blurry mips
  vec3 col = textureGrad(uImage, uv, dFdx(q), dFdy(q)).rgb;
  // Content moved in from beyond the painting's edge (head tracking past the overscan) fades
  // to the background instead of smearing the edge pixels
  vec2 edge = min(uv, 1.0 - uv);
  float inside = smoothstep(-0.04, 0.0, min(edge.x, edge.y));

  if (uShowDepth > 0.0) {
    float lines = smoothstep(0.92, 1.0, fract(hit * 14.0)) * 0.25;
    vec3 dcol = depthPalette(hit) + lines;
    col = mix(col, dcol, uShowDepth);
  }

  if (uFocusFlash > 0.0) {
    float band = 1.0 - smoothstep(0.0, 0.035, abs(hit - uFocus));
    col = mix(col, vec3(0.45, 0.95, 1.0), band * 0.55 * uFocusFlash);
  }

  outColor = vec4(mix(uBackground, col, uFade * inside), 1.0);
}
`;

// Dust motes floating in the light. Each mote lives at a depth, moves with the same
// parallax as the painting, hides behind nearer surfaces and only shows in bright areas.
export const dustVert = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D uImage;
uniform sampler2D uDepth;
uniform float uTime;
uniform float uPointScale;
uniform float uIntensity;

${COMMON}

out float vAlpha;
out float vSoft;
out vec3 vColor;

float hash(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
  float id = float(gl_VertexID);
  float r1 = hash(id * 1.731 + 0.31);
  float r2 = hash(id * 2.113 + 1.71);
  float r3 = hash(id * 3.917 + 4.13);
  float r4 = hash(id * 5.071 + 2.93);

  float z = mix(0.2, 1.0, r3);
  // Slow upward drift with a little sway, wrapping around
  vec2 uv = vec2(r1, r2);
  uv += vec2(0.015 * sin(uTime * (0.2 + 0.2 * r4) + id), -uTime * (0.002 + 0.006 * r4));
  uv = fract(uv);

  vec2 img = qToImg(unproject(uv, z));
  gl_Position = vec4(img.x * 2.0 - 1.0, 1.0 - img.y * 2.0, 0.0, 1.0);

  float surface = textureLod(uDepth, uv, 0.0).r;
  float inFront = smoothstep(0.0, 0.06, z - surface);
  vec3 light = textureLod(uImage, uv, 4.0).rgb;
  float lum = dot(light, vec3(0.299, 0.587, 0.114));
  float lit = smoothstep(0.25, 0.8, lum);
  float defocus = abs(z - uFocus);
  float twinkle = 0.55 + 0.45 * sin(uTime * (0.6 + r4) + id * 7.0);

  gl_PointSize = uPointScale * (1.5 + 5.0 * z * z) * (1.0 + 5.0 * defocus);
  vAlpha = uIntensity * inFront * lit * twinkle / (1.0 + 18.0 * defocus * defocus);
  vSoft = clamp(defocus * 2.5, 0.25, 1.0);
  vColor = mix(vec3(1.0, 0.93, 0.8), light * 1.3 + 0.15, 0.35);
}
`;

export const dustFrag = /* glsl */ `#version 300 es
precision highp float;
in float vAlpha;
in float vSoft;
in vec3 vColor;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  if (r > 1.0) discard;
  float a = (1.0 - smoothstep(1.0 - vSoft, 1.0, r)) * vAlpha;
  outColor = vec4(vColor * a, a); // premultiplied, drawn with additive blending
}
`;
