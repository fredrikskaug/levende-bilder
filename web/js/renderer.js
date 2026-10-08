import { fullscreenVert, paintingFrag, dustVert, dustFrag } from './shaders.js';

const DUST_COUNT = 700;
const MAX_OVERSCAN = 0.12; // share cropped from each side, at most

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
      dust: this.#program(dustVert, dustFrag),
    };
    this.vao = gl.createVertexArray(); // attribute-less draws
    this.aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    this.image = null;
    this.depth = null;
    this.size = [1, 1];
    this.overscan = 0;
    this.background = [0.039, 0.039, 0.043];
  }

  /** Replace the textures with a new painting and its depth map (ImageBitmaps). */
  setPainting(imageBitmap, depthBitmap) {
    const gl = this.gl;
    if (this.image) gl.deleteTexture(this.image);
    if (this.depth) gl.deleteTexture(this.depth);
    this.image = this.#texture(imageBitmap, gl.RGBA8, gl.RGBA, true);
    this.depth = this.#texture(depthBitmap, gl.R8, gl.RED, false);
    this.size = [imageBitmap.width, imageBitmap.height];
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
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.image) return;

    const vy = this.canvas.height - rect.y - rect.h; // GL viewports start bottom-left
    gl.viewport(rect.x, vy, rect.w, rect.h);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(Math.max(0, rect.x), Math.max(0, vy), Math.min(rect.w, this.canvas.width), Math.min(rect.h, this.canvas.height));
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.image);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.depth);

    const common = (prog) => {
      this.#set(prog, 'uImage', 0, 'i');
      this.#set(prog, 'uDepth', 1, 'i');
      this.#set(prog, 'uOverscan', [this.overscan, this.overscan]);
      this.#set(prog, 'uShift', shift);
      this.#set(prog, 'uFocus', p.focus);
      this.#set(prog, 'uDolly', p.dolly);
    };

    const painting = this.programs.painting;
    gl.useProgram(painting.program);
    common(painting);
    this.#set(painting, 'uMode', p.mode, 'i');
    this.#set(painting, 'uSteps', steps, 'i');
    this.#set(painting, 'uShowDepth', p.showDepth);
    this.#set(painting, 'uFocusFlash', p.focusFlash);
    this.#set(painting, 'uFade', p.fade);
    this.#set(painting, 'uBackground', this.background);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (p.dust > 0.001) {
      const dust = this.programs.dust;
      gl.useProgram(dust.program);
      common(dust);
      this.#set(dust, 'uTime', p.time);
      this.#set(dust, 'uPointScale', rect.w / 1400);
      this.#set(dust, 'uIntensity', p.dust * p.fade * (1 - p.showDepth));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArrays(gl.POINTS, 0, DUST_COUNT);
    }

    return { steps, overscan: this.overscan };
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
    else if (typeof value === 'number') gl.uniform1f(loc, value);
    else if (value.length === 2) gl.uniform2fv(loc, value);
    else if (value.length === 3) gl.uniform3fv(loc, value);
  }
}
