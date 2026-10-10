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

// One picture, one depth map: for works without depth layers, and to compare with them
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
uniform bool uDebug;        // mark edges that stretch

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

// For the debug view: did the ray run into the side of something nearer (which stretches)?
// Compares how steeply the surface rises along the ray's way with how fast the ray comes down.
bool stretched(vec2 uv) {
  vec2 size = vec2(textureSize(uDepth, 0));
  vec2 along = -uShift * size;               // pixels the ray moves per unit of depth
  float run = length(along);
  if (run < 1e-3) return false;
  vec2 e = normalize(along) * 2.0 / size;
  float rise = (depthAt(uv + e) - depthAt(uv - e)) / 4.0;
  return rise * run > 2.0;                   // stretched more than 3×
}

void main() {
  vec2 q = imgToQ(vImg);
  float hit;
  vec2 uv = uMode == 0 ? parallaxNaive(q, hit) : parallaxOcclusion(q, hit);

  // Use the undisplaced gradients for mip selection, otherwise depth edges pick blurry mips
  vec3 col = textureGrad(uImage, uv, dFdx(q), dFdy(q)).rgb;
  if (uDebug && uMode == 1 && stretched(uv)) col = mix(col, vec3(1.0, 0.9, 0.0), 0.6);
  // Content moved in from beyond the painting's edge (head tracking past the overscan) fades
  // to the background instead of smearing the edge pixels
  vec2 edge = min(uv, 1.0 - uv);
  float inside = smoothstep(-0.04, 0.0, min(edge.x, edge.y));

  // The depth map as the computer sees it: white near, black far
  if (uShowDepth > 0.0) col = mix(col, vec3(hit), uShowDepth);

  if (uFocusFlash > 0.0) {
    float band = 1.0 - smoothstep(0.0, 0.035, abs(hit - uFocus));
    col = mix(col, vec3(0.45, 0.95, 1.0), band * 0.55 * uFocusFlash);
  }

  outColor = vec4(mix(uBackground, col, uFade * inside), 1.0);
}
`;

// Depth layers (tools/layers.py): each is one smooth surface with its own depth, its own pixels
// (the painting's), what's hidden behind nearer layers (filled in by AI) and nothing elsewhere,
// so its edge is its outline in the alpha map. For each pixel on screen, the view ray is followed
// through the layers front to back: where it meets each layer's surface, and whether the layer has
// anything there. The first that has is shown, and its soft edge is blended with what's behind.
// So an edge is the painting's own outline, moved whole with the layer, at any angle.
export const layersFrag = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2D uImage;          // the painting itself, for the layers' own pixels
uniform sampler2DArray uColors;    // per layer: what AI filled in behind the nearer layers
uniform sampler2DArray uAlphas;    // per layer: 1 its own pixels, 0.86 its rim, 0.63 filled in, 0 nothing
uniform sampler2DArray uDepths;    // per layer: its surface, smooth everywhere
uniform int uCount;                // layers, back to front
uniform vec2 uRange[8];            // each layer's lowest and highest depth
uniform float uTravel;             // texels the view moves per unit of depth, at most
uniform vec2 uRectPx;              // the painting's size on screen, in pixels
uniform float uShowDepth;
uniform float uFocusFlash;
uniform float uFade;
uniform vec3 uBackground;
uniform bool uDebug;               // mark what was filled in by AI

${COMMON}
in vec2 vImg;
out vec4 outColor;

float depthOf(int k, vec2 uv) { return textureLod(uDepths, vec3(uv, float(k)), 0.0).r; }

// Where the view ray through q meets layer k's surface: marched from the layer's nearest depth
// back to its farthest, in steps of about a texel, with a linear refinement at the crossing.
vec2 meet(vec2 q, int k, out float hit) {
  vec2 range = uRange[k];
  int steps = clamp(int(ceil((range.y - range.x) * uTravel)), 1, 48);
  float stepSize = (range.y - range.x) / float(steps);
  float z = range.y;
  float h = depthOf(k, project(q, z));
  float prevZ = z;
  float prevH = h;
  for (int i = 0; i < 48; i++) {
    if (h >= z || i >= steps) break;
    prevZ = z;
    prevH = h;
    z -= stepSize;
    h = depthOf(k, project(q, z));
  }
  float before = prevZ - prevH;
  float after = z - h;
  float t = clamp(before / max(before - after, 1e-5), 0.0, 1.0);
  hit = mix(prevZ, z, t);
  return project(q, hit);
}

// Layer k at uv, from the four texels around it like a bilinear lookup, but counting only the
// texels the layer has: how much of the pixel it covers, and its colour there, the painting's
// where it has its own texels, else what was filled in. Filled-in texels (under a nearer layer)
// count only towards what the nearer layers haven't covered yet (\`covered\`): so at rest a soft
// edge is made of the painting's own pixels exactly as they are, never of what's hidden under the
// nearer one. The layer's rim, taken over from what's behind, counts as much as \`keep\`.
float sampleLayer(int k, vec2 uv, float keep, float covered, vec2 gx, vec2 gy, out vec3 colour, out bool filledIn) {
  ivec2 size = textureSize(uAlphas, 0).xy;
  vec2 st = uv * vec2(size) - 0.5;
  ivec2 i0 = ivec2(floor(st));
  vec2 f = st - floor(st);
  float ownW = 0.0;
  float fillW = 0.0;   // filled in, under a nearer layer
  vec3 ownC = vec3(0.0);
  vec3 fillC = vec3(0.0);
  bool inside = true;  // all four its own, nothing else involved
  for (int j = 0; j < 4; j++) {
    ivec2 o = ivec2(j & 1, j >> 1);
    ivec2 t = clamp(i0 + o, ivec2(0), size - 1);
    float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
    float a = texelFetch(uAlphas, ivec3(t, k), 0).r;
    inside = inside && a > 0.97;
    if (a < 0.3) continue;
    if (a < 0.8) {
      fillW += w;
      fillC += w * texelFetch(uColors, ivec3(t, k), 0).rgb;
    } else {
      if (a < 0.97) w *= keep;
      ownW += w;
      ownC += w * texelFetch(uImage, t, 0).rgb;
    }
  }
  float s = fillW > 0.0 ? clamp((fillW - covered) / fillW, 0.0, 1.0) : 1.0;
  fillW *= s;
  fillC *= s;
  filledIn = ownW <= 0.0;
  // Inside, filtered as usual (with mipmaps when the painting is shown smaller than it is)
  colour = inside ? textureGrad(uImage, uv, gx, gy).rgb : ownW > 0.0 ? ownC / ownW : fillC / max(fillW, 1e-6);
  return ownW + fillW;
}

void main() {
  vec2 q = imgToQ(vImg);
  vec2 gx = dFdx(q);
  vec2 gy = dFdy(q);
  float moved = length(uShift * uRectPx);  // screen pixels per unit of depth difference
  vec3 col = vec3(0.0);
  float covered = 0.0;
  float shownDepth = 0.0;
  vec2 shownUv = q;
  vec2 lastUv = q;
  float lastHit = 0.0;
  for (int k = 7; k >= 0; k--) {
    if (k >= uCount) continue;
    float hit;
    vec2 uv = meet(q, k, hit);
    lastUv = uv;
    lastHit = hit;
    // The rim a nearer layer took over from what's behind (the depth map's soft edge, in the
    // background's colours): let it go as the layer moves away from what's behind, so it doesn't
    // drag a halo along. At rest it stays, and the picture is the painting.
    float keep = k > 0 ? 1.0 - smoothstep(0.5, 2.0, moved * max(hit - depthOf(k - 1, uv), 0.0)) : 1.0;
    vec3 c;
    bool filledIn;
    float coverage = sampleLayer(k, uv, keep, covered, gx, gy, c, filledIn);
    if (coverage <= 0.0) continue;
    if (uDebug && filledIn) c = mix(c, vec3(1.0, 0.0, 0.8), 0.55);
    // It fills what the nearer layers left uncovered, as far as it covers: at a soft edge the
    // layers' texels are side by side, not on top of each other
    float w = min(coverage, 1.0 - covered);
    if (covered == 0.0) { shownDepth = hit; shownUv = uv; }
    col += w * c;
    covered += w;
    if (covered > 0.996) break;
  }
  // Where no layer has anything (beyond what was filled in), the farthest layer's surface,
  // stretched as before, instead of a hole
  if (covered < 0.996) {
    vec3 c = textureGrad(uImage, lastUv, gx, gy).rgb;
    if (uDebug) c = mix(c, vec3(1.0, 0.9, 0.0), 0.6);
    if (covered == 0.0) { shownDepth = lastHit; shownUv = lastUv; }
    col += (1.0 - covered) * c;
  }

  // Content moved in from beyond the painting's edge fades to the background
  vec2 edge = min(shownUv, 1.0 - shownUv);
  float inside = smoothstep(-0.04, 0.0, min(edge.x, edge.y));
  // The depth as the computer sees it: white near, black far
  if (uShowDepth > 0.0) col = mix(col, vec3(shownDepth), uShowDepth);
  if (uFocusFlash > 0.0) {
    float band = 1.0 - smoothstep(0.0, 0.035, abs(shownDepth - uFocus));
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
