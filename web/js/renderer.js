import { fullscreenVert, paintingFrag, layersFrag, dustVert, dustFrag } from './shaders.js';

const DUST_COUNT = 700;
const MAX_OVERSCAN = 0.12; // share cropped from each side, at most
const MAX_LAYERS = 8;

/** Lowest and highest value in a greyscale ImageBitmap, 0..1. */
function valueRange(bitmap) {
  const { width: w, height: h } = bitmap;
  const canvas = typeof OffscreenCanvas === 'undefined'
    ? Object.assign(document.createElement('canvas'), { width: w, height: h })
    : new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  let lo = 255;
  let hi = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] < lo) lo = data[i];
    if (data[i] > hi) hi = data[i];
  }
  return [Math.max(0, lo - 1) / 255, Math.min(255, hi + 1) / 255];
}

/**
 * Draws one painting with depth-based parallax (+ optional dust) into a WebGL2 canvas.
 * Everything is drawn with the viewport set to the painting's rectangle, so the shaders
 * work directly in painting space.
 */
export class ParallaxRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('Nettleseren din støtter ikke WebGL2.');
    this.gl = gl;
    this.programs = {
      painting: this.#program(fullscreenVert, paintingFrag),
      layers: this.#program(fullscreenVert, layersFrag),
      dust: this.#program(dustVert, dustFrag),
    };
    this.vao = gl.createVertexArray(); // attribute-less draws
    this.aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    this.image = null;
    this.depth = null;
    this.layers = null;  // depth layers (tools/layers.py): { count, colors, alphas, depths, ranges }
    this.size = [1, 1];
    this.overscan = 0;
    this.background = [0.039, 0.039, 0.043];
  }

  /**
   * Replace the textures with a new painting and its depth map (ImageBitmaps), and its depth
   * layers if it has them: [{ color, alpha, depth }] back to front.
   */
  setPainting(imageBitmap, depthBitmap, layers = null) {
    const gl = this.gl;
    const old = [this.image, this.depth, this.layers?.colors, this.layers?.alphas, this.layers?.depths];
    for (const tex of old) if (tex) gl.deleteTexture(tex);
    this.image = this.#texture(imageBitmap, gl.RGBA8, gl.RGBA, true);
    this.depth = this.#texture(depthBitmap, gl.R8, gl.RED, false);
    this.size = [imageBitmap.width, imageBitmap.height];
    this.layers = null;
    if (layers?.length) {
      const stack = layers.slice(-MAX_LAYERS);
      this.layers = {
        count: stack.length,
        colors: this.#array(stack.map((l) => l.color), gl.RGBA8, gl.RGBA),
        alphas: this.#array(stack.map((l) => l.alpha), gl.R8, gl.RED),
        depths: this.#array(stack.map((l) => l.depth), gl.R8, gl.RED),
        ranges: new Float32Array(MAX_LAYERS * 2),
      };
      stack.forEach((l, k) => this.layers.ranges.set(valueRange(l.depth), 2 * k));
    }
  }

  get hasLayers() {
    return Boolean(this.layers);
  }

  /** The layers' maps as one texture array (slices of another size are left empty). */
  #array(bitmaps, internalFormat, format) {
    const gl = this.gl;
    const [w, h] = this.size;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internalFormat, w, h, bitmaps.length);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    bitmaps.forEach((b, k) => {
      if (b.width === w && b.height === h) gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, k, w, h, 1, format, gl.UNSIGNED_BYTE, b);
    });
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    return tex;
  }

  /**
   * Painting rectangle in device pixels, top-left origin.
   * contain: fit inside `area` (device px), fill: as large as possible on the whole canvas
   * without cropping, cover: fill the whole canvas, cropping what doesn't fit (video clips).
   */
  layout(fit, area) {
    const [iw, ih] = this.size;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const box = fit === 'contain' ? area : { x: 0, y: 0, w: cw, h: ch };
    const scale = fit === 'cover' ? Math.max(box.w / iw, box.h / ih) : Math.min(box.w / iw, box.h / ih);
    const w = Math.round(iw * scale);
    const h = Math.round(ih * scale);
    return { x: Math.round(box.x + (box.w - w) / 2), y: Math.round(box.y + (box.h - h) / 2), w, h };
  }

  /**
   * @param {object} p
   * @param {{x,y,w,h}} p.rect      painting rectangle from layout()
   * @param {number[]} p.shift      camera offset: how far the farthest/nearest layer moves relative
   *                                to the focus plane, as a fraction of the painting width (both axes)
   * @param {number} p.focus        depth (0 far – 1 near) that stays still
   * @param {number} p.dolly        dolly zoom amount
   * @param {number[]} p.shiftRange |shift| on each axis to hide behind the overscan; larger shifts
   *                                fade to black at the edges instead of cropping more
   * @param {number} p.dollyRange   largest |dolly| expected
   * @param {number} p.mode         0 = naive offset, 1 = parallax occlusion mapping
   * @param {number} p.maxSteps
   * @param {number} p.showDepth    0..1
   * @param {number} p.focusFlash   0..1
   * @param {number} p.fade         0..1
   * @param {number} p.dust         0..1 dust intensity
   * @param {number} p.time         seconds
   * @param {boolean} p.layered     draw the depth layers, if the work has them (mode 1 only)
   * @param {boolean} p.debug       mark what was filled in by AI (magenta), or edges that stretch (yellow)
   */
  render(p) {
    const gl = this.gl;
    const { rect } = p;
    const [iw, ih] = this.size;
    const aspect = iw / ih;
    const reach = Math.max(p.focus, 1 - p.focus);

    // Same zoom on both axes keeps the aspect ratio; ease it so toggling dolly doesn't pop.
    // Capped: beyond it the edges fade rather than the whole picture zooming in.
    const [rx, ry] = p.shiftRange;
    const ox = rx * reach + 0.5 * p.dollyRange * reach;
    const oy = ry * aspect * reach + 0.5 * p.dollyRange * reach;
    const target = Math.min(MAX_OVERSCAN, Math.max(ox, oy) + 0.002);
    this.overscan += (target - this.overscan) * 0.08;
    if (Math.abs(target - this.overscan) < 1e-4) this.overscan = target;

    // p.shift is in widths of the painting as shown; the overscan zooms the texture in, so the
    // same move on screen is a smaller one in the texture. uv: y is in units of the painting height.
    const zoom = 1 - 2 * this.overscan;
    const shift = [p.shift[0] * zoom, p.shift[1] * aspect * zoom];
    // Enough ray-march steps that each step moves ~2 texels across the full depth range. Sized
    // from the *range*, not the current shift: a step count that changes while the camera moves
    // makes depth edges pop by a pixel each time.
    const travel =
      Math.hypot(Math.max(rx, Math.abs(p.shift[0])) * iw, Math.max(ry, Math.abs(p.shift[1])) * iw) +
      0.5 * Math.max(p.dollyRange, Math.abs(p.dolly)) * iw;
    const steps = Math.max(8, Math.min(p.maxSteps, Math.ceil(travel / 2)));

    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(...this.background, 1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.image) return;

    const vy = this.canvas.height - rect.y - rect.h; // GL viewports start bottom-left
    gl.viewport(rect.x, vy, rect.w, rect.h);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(Math.max(0, rect.x), Math.max(0, vy), Math.min(rect.w, this.canvas.width), Math.min(rect.h, this.canvas.height));
    const bindPainting = () => {
      gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.image);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.depth);
    };
    const layered = Boolean(p.layered && p.mode === 1 && this.layers);

    const common = (prog) => {
      this.#set(prog, 'uImage', 0, 'i');
      this.#set(prog, 'uDepth', 1, 'i');
      this.#set(prog, 'uOverscan', [this.overscan, this.overscan]);
      this.#set(prog, 'uShift', shift);
      this.#set(prog, 'uFocus', p.focus);
      this.#set(prog, 'uDolly', p.dolly);
    };

    gl.disable(gl.BLEND);
    if (layered) {
      // All layers in one pass: per pixel, the first layer the view ray meets that has anything there
      const prog = this.programs.layers;
      bindPainting();
      gl.useProgram(prog.program);
      common(prog);
      const units = [['uColors', this.layers.colors], ['uAlphas', this.layers.alphas], ['uDepths', this.layers.depths]];
      units.forEach(([name, tex], i) => {
        gl.activeTexture(gl.TEXTURE2 + i);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
        this.#set(prog, name, 2 + i, 'i');
      });
      this.#set(prog, 'uCount', this.layers.count, 'i');
      this.#set(prog, 'uRange', this.layers.ranges, 'v2');
      this.#set(prog, 'uTravel', travel);
      this.#set(prog, 'uRectPx', [rect.w, rect.h]);
      this.#set(prog, 'uShowDepth', p.showDepth);
      this.#set(prog, 'uFocusFlash', p.focusFlash);
      this.#set(prog, 'uFade', p.fade);
      this.#set(prog, 'uBackground', this.background);
      this.#set(prog, 'uDebug', p.debug ? 1 : 0, 'i');
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      const painting = this.programs.painting;
      bindPainting();
      gl.useProgram(painting.program);
      common(painting);
      this.#set(painting, 'uMode', p.mode, 'i');
      this.#set(painting, 'uSteps', steps, 'i');
      this.#set(painting, 'uShowDepth', p.showDepth);
      this.#set(painting, 'uFocusFlash', p.focusFlash);
      this.#set(painting, 'uFade', p.fade);
      this.#set(painting, 'uBackground', this.background);
      this.#set(painting, 'uDebug', p.debug ? 1 : 0, 'i');
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    if (p.dust > 0.001) {
      const dust = this.programs.dust;
      bindPainting();
      gl.useProgram(dust.program);
      common(dust);
      this.#set(dust, 'uTime', p.time);
      this.#set(dust, 'uPointScale', rect.w / 1400);
      this.#set(dust, 'uIntensity', p.dust * p.fade * (1 - p.showDepth));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArrays(gl.POINTS, 0, DUST_COUNT);
    }

    return { steps: layered ? `${this.layers.count} lag` : steps, overscan: this.overscan };
  }

  #texture(source, internalFormat, format, mipmaps) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // row 0 = top of the image = v 0
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, format, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    if (mipmaps) {
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      if (this.aniso) gl.texParameterf(gl.TEXTURE_2D, this.aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
    } else {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    return tex;
  }

  #program(vsSource, fsSource) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return { program, locations: new Map() };
  }

  #set(prog, name, value, type = 'f') {
    const gl = this.gl;
    let loc = prog.locations.get(name);
    if (loc === undefined) {
      loc = gl.getUniformLocation(prog.program, name);
      prog.locations.set(name, loc);
    }
    if (loc === null) return;
    if (type === 'i') gl.uniform1i(loc, value);
    else if (type === 'v2') gl.uniform2fv(loc, value);
    else if (typeof value === 'number') gl.uniform1f(loc, value);
    else if (value.length === 2) gl.uniform2fv(loc, value);
    else if (value.length === 3) gl.uniform3fv(loc, value);
  }
}
