/**
 * LYNX DFT 3D Viewer
 *
 * - PBR atoms with element-specific metalness, roughness, clearcoat
 * - Ray-marched volumetric density (hydrogen orbital cloud style)
 * - Cross-section slice planes
 * - Marching-cubes isosurface
 * - Camera viewpoint presets with smooth animation
 */

// ═════════════════════════════════════════════════════════════════════════
//  Colormaps  (256-entry LUTs, [r,g,b] in 0-1)
// ═════════════════════════════════════════════════════════════════════════
const Colormaps = (() => {
  function lerp(a, b, t) { return a + (b - a) * t; }
  function buildFromStops(stops) {
    const N = 256, out = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      let s0 = stops[0], s1 = stops[stops.length - 1];
      for (let j = 0; j < stops.length - 1; j++) {
        if (t >= stops[j].pos && t <= stops[j + 1].pos) { s0 = stops[j]; s1 = stops[j + 1]; break; }
      }
      const lt = (t - s0.pos) / (s1.pos - s0.pos || 1);
      out[i*3]   = lerp(s0.r, s1.r, lt);
      out[i*3+1] = lerp(s0.g, s1.g, lt);
      out[i*3+2] = lerp(s0.b, s1.b, lt);
    }
    return out;
  }
  const viridis = buildFromStops([
    {pos:0, r:0.267,g:0.004,b:0.329},{pos:0.25, r:0.282,g:0.140,b:0.458},
    {pos:0.5, r:0.127,g:0.566,b:0.551},{pos:0.75, r:0.544,g:0.774,b:0.248},
    {pos:1, r:0.993,g:0.906,b:0.144},
  ]);
  const plasma = buildFromStops([
    {pos:0, r:0.050,g:0.030,b:0.528},{pos:0.25, r:0.494,g:0.012,b:0.658},
    {pos:0.5, r:0.798,g:0.280,b:0.470},{pos:0.75, r:0.973,g:0.585,b:0.254},
    {pos:1, r:0.940,g:0.975,b:0.131},
  ]);
  const inferno = buildFromStops([
    {pos:0, r:0.001,g:0.000,b:0.014},{pos:0.25, r:0.341,g:0.062,b:0.429},
    {pos:0.5, r:0.735,g:0.216,b:0.330},{pos:0.75, r:0.973,g:0.557,b:0.055},
    {pos:1, r:0.988,g:0.998,b:0.645},
  ]);
  const coolwarm = buildFromStops([
    {pos:0, r:0.230,g:0.299,b:0.754},{pos:0.5, r:0.865,g:0.865,b:0.865},
    {pos:1, r:0.706,g:0.016,b:0.150},
  ]);
  const blueRed = buildFromStops([
    {pos:0, r:0.0,g:0.0,b:0.5},{pos:0.25, r:0.0,g:0.3,b:1.0},
    {pos:0.5, r:0.0,g:1.0,b:0.5},{pos:0.75, r:1.0,g:1.0,b:0.0},
    {pos:1, r:1.0,g:0.0,b:0.0},
  ]);
  // Orbital-style: deep blue → cyan → white glow
  const orbital = buildFromStops([
    {pos:0,    r:0.02, g:0.01, b:0.15},
    {pos:0.15, r:0.05, g:0.05, b:0.45},
    {pos:0.35, r:0.10, g:0.30, b:0.80},
    {pos:0.55, r:0.25, g:0.65, b:0.95},
    {pos:0.75, r:0.60, g:0.90, b:1.00},
    {pos:0.90, r:0.90, g:0.97, b:1.00},
    {pos:1,    r:1.00, g:1.00, b:1.00},
  ]);
  function sample(cmap, t) {
    const idx = Math.max(0, Math.min(255, Math.floor(t * 255)));
    return [cmap[idx*3], cmap[idx*3+1], cmap[idx*3+2]];
  }
  return { viridis, plasma, inferno, coolwarm, blueRed, orbital, sample };
})();

// ═════════════════════════════════════════════════════════════════════════
//  Per-element PBR material properties
// ═════════════════════════════════════════════════════════════════════════
const ELEM_PBR = {
  // Metals — high metalness, low roughness, strong clearcoat
  'Li': {m:0.6, r:0.25, cc:0.8,  ccr:0.1},
  'Na': {m:0.7, r:0.20, cc:0.9,  ccr:0.08},
  'K':  {m:0.7, r:0.22, cc:0.8,  ccr:0.10},
  'Ca': {m:0.5, r:0.30, cc:0.7,  ccr:0.15},
  'Ti': {m:0.85,r:0.12, cc:0.9,  ccr:0.05},
  'Fe': {m:0.90,r:0.10, cc:0.6,  ccr:0.10},
  'Cu': {m:0.95,r:0.08, cc:0.9,  ccr:0.04},
  'Zn': {m:0.7, r:0.18, cc:0.7,  ccr:0.10},
  'Au': {m:1.0, r:0.05, cc:1.0,  ccr:0.02},
  'Al': {m:0.8, r:0.10, cc:0.95, ccr:0.03},
  'Ba': {m:0.6, r:0.20, cc:0.85, ccr:0.08},
  // Semiconductors — mid metalness, glossy
  'Si': {m:0.45,r:0.18, cc:1.0,  ccr:0.05},
  'Ge': {m:0.50,r:0.15, cc:1.0,  ccr:0.05},
  'Ga': {m:0.55,r:0.14, cc:0.9,  ccr:0.06},
  'As': {m:0.50,r:0.16, cc:0.85, ccr:0.08},
  // Non-metals — low metalness, glassy
  'H':  {m:0.0, r:0.05, cc:1.0,  ccr:0.02},
  'C':  {m:0.05,r:0.30, cc:0.6,  ccr:0.15},
  'N':  {m:0.0, r:0.10, cc:1.0,  ccr:0.04},
  'O':  {m:0.0, r:0.08, cc:1.0,  ccr:0.03},
  'F':  {m:0.0, r:0.06, cc:1.0,  ccr:0.02},
  'S':  {m:0.05,r:0.12, cc:0.9,  ccr:0.05},
  'Cl': {m:0.0, r:0.08, cc:1.0,  ccr:0.03},
  'P':  {m:0.1, r:0.15, cc:0.8,  ccr:0.08},
  // Noble gases — perfectly transparent glass
  'He': {m:0.0, r:0.02, cc:1.0,  ccr:0.01},
  'Ne': {m:0.0, r:0.02, cc:1.0,  ccr:0.01},
  'Ar': {m:0.0, r:0.03, cc:1.0,  ccr:0.01},
};
const ELEM_PBR_DEFAULT = {m:0.3, r:0.20, cc:0.8, ccr:0.08};


// ═════════════════════════════════════════════════════════════════════════
//  Volume-rendering GLSL shaders (WebGL 2 / GLSL 300 es)
// ═════════════════════════════════════════════════════════════════════════
const VOL_VERT = `#version 300 es
precision highp float;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPosition;

in vec3 position;

out vec3 vOrigin;   // camera pos in local [0,1]^3 space
out vec3 vDirection; // direction from camera to vertex in local space

uniform mat4 uInvModel;

void main() {
  vOrigin    = (uInvModel * vec4(cameraPosition, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const VOL_FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler2D uColormap;
uniform float uThreshold;   // 0-1 fraction below which density is invisible
uniform float uOpacity;     // global opacity multiplier
uniform float uBrightness;  // glow intensity
uniform int   uSteps;       // ray march steps

vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = -o * inv;
  vec3 t1 = (1.0 - o) * inv;
  vec3 mn = min(t0, t1);
  vec3 mx = max(t0, t1);
  float tN = max(max(mn.x, mn.y), mn.z);
  float tF = min(min(mx.x, mx.y), mx.z);
  return vec2(tN, tF);
}

void main() {
  vec3 rd = normalize(vDirection);
  vec2 tb = hitBox(vOrigin, rd);
  if (tb.x > tb.y) discard;
  tb.x = max(tb.x, 0.0);

  float dt = (tb.y - tb.x) / float(uSteps);
  vec4 acc = vec4(0.0);

  for (int i = 0; i < 256; i++) {
    if (i >= uSteps) break;

    vec3 p = vOrigin + rd * (tb.x + float(i) * dt + dt * 0.5);
    if (p.x < 0.0 || p.y < 0.0 || p.z < 0.0 ||
        p.x > 1.0 || p.y > 1.0 || p.z > 1.0) continue;

    float d = texture(uVolume, p).r;

    if (d > uThreshold) {
      float t  = (d - uThreshold) / (1.0 - uThreshold);
      t = clamp(t, 0.0, 1.0);

      // Sample colormap
      vec3 col = texture(uColormap, vec2(pow(t, 0.55), 0.5)).rgb;

      // Boost brightness at high density (glow)
      col *= 1.0 + t * uBrightness;

      // Opacity ramp: smooth entry, quadratic growth
      float a = smoothstep(0.0, 0.08, t) * t * uOpacity * dt * 28.0;

      // Front-to-back compositing
      acc.rgb += (1.0 - acc.a) * a * col;
      acc.a   += (1.0 - acc.a) * a;

      if (acc.a > 0.96) break;
    }
  }

  fragColor = acc;
}
`;


// ═════════════════════════════════════════════════════════════════════════
//  LYNXViewer
// ═════════════════════════════════════════════════════════════════════════
class LYNXViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);

    // ── Renderer ──
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x06080c);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    this.scene  = new THREE.Scene();

    // ── Camera ──
    this.camera = new THREE.PerspectiveCamera(
      45, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
    this.camera.position.set(20, 15, 20);

    // ── Controls ──
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 200;

    // ── Lighting (studio-style for PBR) ──
    // Hemisphere: sky / ground
    const hemi = new THREE.HemisphereLight(0x94b0d4, 0x262830, 0.45);
    this.scene.add(hemi);

    // Key light
    const key = new THREE.DirectionalLight(0xfff4e6, 1.0);
    key.position.set(12, 25, 15);
    this.scene.add(key);

    // Fill light
    const fill = new THREE.DirectionalLight(0xb0c4de, 0.45);
    fill.position.set(-15, 5, -10);
    this.scene.add(fill);

    // Rim / back light
    const rim = new THREE.DirectionalLight(0xaaccff, 0.35);
    rim.position.set(0, -8, -20);
    this.scene.add(rim);

    // ── Environment map for PBR reflections ──
    this._envMap = null;
    this._buildEnvMap();

    // ── Groups ──
    this.atomGroup    = new THREE.Group();
    this.bondGroup    = new THREE.Group();
    this.cellGroup    = new THREE.Group();
    this.densityGroup = new THREE.Group(); // isosurface
    this.volumeGroup  = new THREE.Group(); // ray-marched volume
    this.sliceGroup   = new THREE.Group();
    this.labelGroup   = new THREE.Group();

    [this.atomGroup, this.bondGroup, this.cellGroup,
     this.densityGroup, this.volumeGroup, this.sliceGroup, this.labelGroup
    ].forEach(g => this.scene.add(g));

    // ── State ──
    this.atoms = [];
    this.cell  = null;
    this.densityData   = null;
    this.densityMesh   = null;
    this.volumeBox     = null;
    this.sliceMeshes   = {x:null, y:null, z:null};
    this.slicePositions= {x:0.5, y:0.5, z:0.5};

    this.isovalue      = 0.05;
    this.densityOpacity= 0.6;
    this.densityColor  = 0x4488ff;

    this.volumeThresholdLow = 0.08;
    this.volumeOpacity  = 0.7;
    this.volumeBrightness = 1.8;
    this.volumeSteps    = 160;
    this.volumePointSize= 3.0;
    this.currentColormap= 'orbital';
    this.smoothNormals  = true;

    // Shared geometries
    this._sphereGeo   = new THREE.SphereBufferGeometry(1, 48, 32);
    this._cylinderGeo = new THREE.CylinderBufferGeometry(1, 1, 1, 12);

    this.elementData = {};
    this._cellCenter = new THREE.Vector3();
    this._cellRadius = 15;
    this._invModel   = new THREE.Matrix4();

    // ── Resize ──
    this._onResize = () => {
      const w = this.container.clientWidth, h = this.container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    };
    window.addEventListener('resize', this._onResize);

    this._animate();
  }

  // ── Environment map for atom reflections ──────────────────────────────
  _buildEnvMap() {
    try {
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      pmrem.compileEquirectangularShader();
      const envScene = new THREE.Scene();
      envScene.background = new THREE.Color(0x0c1018);
      const l1 = new THREE.PointLight(0xffffff, 200, 0);
      l1.position.set(15, 20, 10);
      envScene.add(l1);
      const l2 = new THREE.PointLight(0x6688cc, 80, 0);
      l2.position.set(-12, 8, -8);
      envScene.add(l2);
      const l3 = new THREE.PointLight(0xcc8866, 60, 0);
      l3.position.set(5, -10, 15);
      envScene.add(l3);
      this._envMap = pmrem.fromScene(envScene, 0, 0.1, 100).texture;
      this.scene.environment = this._envMap;
      pmrem.dispose();
    } catch(e) {
      console.warn('PMREMGenerator not available, skipping env map', e);
    }
  }

  // ── Render loop ───────────────────────────────────────────────────────
  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    // Update volume shader uniforms (camera position changes every frame)
    if (this.volumeBox && this.volumeBox.material.uniforms) {
      this._invModel.copy(this.volumeBox.matrixWorld).invert();
      this.volumeBox.material.uniforms.uInvModel.value.copy(this._invModel);
    }

    this.renderer.render(this.scene, this.camera);
  }

  setElementData(data) { this.elementData = data; }

  // ═════════════════════════════════════════════════════════════════════
  //  STRUCTURE
  // ═════════════════════════════════════════════════════════════════════

  loadStructure(config) {
    const lv = config.lattice.vectors;
    this.cell = lv;
    this.atoms = [];
    for (const g of config.atoms) {
      const el = g.element, frac = g.fractional !== false;
      for (const c of g.coordinates) {
        const pos = frac
          ? [c[0]*lv[0][0]+c[1]*lv[1][0]+c[2]*lv[2][0],
             c[0]*lv[0][1]+c[1]*lv[1][1]+c[2]*lv[2][1],
             c[0]*lv[0][2]+c[1]*lv[1][2]+c[2]*lv[2][2]]
          : c.slice();
        this.atoms.push({element:el, position:pos, frac: frac ? c : null});
      }
    }
    this._buildAtoms();
    this._buildBonds();
    this._buildCell();
    this._computeCellCenter();
    this._centerCamera();
  }

  loadFromResults(sys) {
    this.cell = sys.lattice_vectors;
    this.atoms = sys.atoms.map(a => ({element:a.element, position:a.position_cart, frac:a.position_frac}));
    this._buildAtoms(); this._buildBonds(); this._buildCell();
    this._computeCellCenter(); this._centerCamera();
  }

  // ── PBR Atoms ─────────────────────────────────────────────────────────
  _getElemColor(elem) {
    const d = this.elementData[elem];
    return d ? new THREE.Color(d.color) : new THREE.Color(0xff69b4);
  }
  _getElemRadius(elem) {
    const d = this.elementData[elem];
    return d ? d.radius * 0.45 : 0.55;
  }

  _buildAtoms() {
    this.atomGroup.clear();
    for (const atom of this.atoms) {
      const pbr  = ELEM_PBR[atom.element] || ELEM_PBR_DEFAULT;
      const col  = this._getElemColor(atom.element);
      const rad  = this._getElemRadius(atom.element);

      const mat = new THREE.MeshPhysicalMaterial({
        color: col,
        metalness: pbr.m,
        roughness: pbr.r,
        clearcoat: pbr.cc,
        clearcoatRoughness: pbr.ccr,
        reflectivity: 0.6,
        envMapIntensity: 1.2,
      });

      const mesh = new THREE.Mesh(this._sphereGeo, mat);
      mesh.scale.setScalar(rad);
      mesh.position.set(...atom.position);
      mesh.userData = {element: atom.element};
      this.atomGroup.add(mesh);
    }
  }

  // ── Bonds ─────────────────────────────────────────────────────────────
  _buildBonds() {
    this.bondGroup.clear();
    if (this.atoms.length < 2) return;
    const CUT = 3.5;
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x8899aa,
      metalness: 0.4,
      roughness: 0.35,
      clearcoat: 0.5,
      clearcoatRoughness: 0.2,
    });
    for (let i = 0; i < this.atoms.length; i++) {
      for (let j = i + 1; j < this.atoms.length; j++) {
        const pi = this.atoms[i].position, pj = this.atoms[j].position;
        const dx = pj[0]-pi[0], dy = pj[1]-pi[1], dz = pj[2]-pi[2];
        const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
        if (d < CUT && d > 0.1) {
          const m = new THREE.Mesh(this._cylinderGeo, mat);
          m.position.set((pi[0]+pj[0])/2,(pi[1]+pj[1])/2,(pi[2]+pj[2])/2);
          m.scale.set(0.055, d, 0.055);
          m.quaternion.setFromUnitVectors(
            new THREE.Vector3(0,1,0),
            new THREE.Vector3(dx,dy,dz).normalize());
          this.bondGroup.add(m);
        }
      }
    }
  }

  // ── Unit Cell ─────────────────────────────────────────────────────────
  _buildCell() {
    this.cellGroup.clear();
    if (!this.cell) return;
    const a=this.cell[0], b=this.cell[1], c=this.cell[2], o=[0,0,0];
    const cn = [o,a,[a[0]+b[0],a[1]+b[1],a[2]+b[2]],b,
      c,[a[0]+c[0],a[1]+c[1],a[2]+c[2]],
      [a[0]+b[0]+c[0],a[1]+b[1]+c[1],a[2]+b[2]+c[2]],
      [b[0]+c[0],b[1]+c[1],b[2]+c[2]]];
    const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const lm = new THREE.LineBasicMaterial({color:0x3a7bd5, opacity:0.45, transparent:true});
    for (const [i,j] of edges) {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...cn[i]), new THREE.Vector3(...cn[j])]);
      this.cellGroup.add(new THREE.Line(g,lm));
    }
    this._addAxisLabel('a', cn[1], 0x58a6ff);
    this._addAxisLabel('b', cn[3], 0x3fb950);
    this._addAxisLabel('c', cn[4], 0xf85149);
  }
  _addAxisLabel(txt, pos, color) {
    const cv = document.createElement('canvas');
    cv.width=64; cv.height=32;
    const cx = cv.getContext('2d');
    cx.fillStyle='#'+color.toString(16).padStart(6,'0');
    cx.font='bold 24px sans-serif'; cx.textAlign='center';
    cx.fillText(txt,32,24);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(cv), transparent:true}));
    sp.position.set(pos[0]*1.06, pos[1]*1.06+0.5, pos[2]*1.06);
    sp.scale.set(2,1,1);
    this.cellGroup.add(sp);
  }

  _computeCellCenter() {
    if (!this.cell) return;
    const a=this.cell[0],b=this.cell[1],c=this.cell[2];
    this._cellCenter.set((a[0]+b[0]+c[0])/2,(a[1]+b[1]+c[1])/2,(a[2]+b[2]+c[2])/2);
    this._cellRadius = Math.max(
      Math.hypot(a[0],a[1],a[2]),
      Math.hypot(b[0],b[1],b[2]),
      Math.hypot(c[0],c[1],c[2]));
  }
  _centerCamera() {
    if (!this.cell) return;
    const r = this._cellRadius;
    this.controls.target.copy(this._cellCenter);
    this.camera.position.set(
      this._cellCenter.x+r*1.2, this._cellCenter.y+r*0.8, this._cellCenter.z+r*1.2);
    this.controls.update();
    this._buildGridFloor();
  }

  // ═════════════════════════════════════════════════════════════════════
  //  CAMERA PRESETS
  // ═════════════════════════════════════════════════════════════════════
  setCameraView(preset) {
    const c = this._cellCenter, r = this._cellRadius * 1.8;
    const V = {
      '+x':[c.x+r,c.y,c.z],  '-x':[c.x-r,c.y,c.z],
      '+y':[c.x,c.y+r,c.z],  '-y':[c.x,c.y-r,c.z],
      '+z':[c.x,c.y,c.z+r],  '-z':[c.x,c.y,c.z-r],
      'iso1':[c.x+r*.7,c.y+r*.7,c.z+r*.7],
      'iso2':[c.x-r*.7,c.y+r*.7,c.z+r*.7],
      'iso3':[c.x+r*.7,c.y+r*.7,c.z-r*.7],
      'iso4':[c.x-r*.7,c.y-r*.7,c.z+r*.7],
    };
    const tgt = V[preset]; if (!tgt) return;
    const s = this.camera.position.clone();
    const e = new THREE.Vector3(...tgt);
    const t0 = performance.now();
    const step = () => {
      const t = Math.min(1,(performance.now()-t0)/400);
      const ease = 1 - Math.pow(1-t,3);
      this.camera.position.lerpVectors(s,e,ease);
      this.controls.target.copy(c);
      this.controls.update();
      if (t<1) requestAnimationFrame(step);
    };
    step();
  }

  // ═════════════════════════════════════════════════════════════════════
  //  DENSITY — Load
  // ═════════════════════════════════════════════════════════════════════
  loadDensity(data) {
    const bytes = Uint8Array.from(atob(data.data_base64), c => c.charCodeAt(0));
    const floats = new Float32Array(bytes.buffer);
    this.densityData = {
      values: floats, shape: data.shape, cell: data.cell,
      min: data.min, max: data.max,
    };
    this.isovalue = data.min + (data.max - data.min) * 0.3;
    this.updateIsosurface();
    this._buildVolumeRenderer();
    this._buildSlicePlanes();
  }

  // ═════════════════════════════════════════════════════════════════════
  //  DENSITY — Isosurface (marching cubes)
  // ═════════════════════════════════════════════════════════════════════
  updateIsosurface() {
    if (!this.densityData) return;
    if (this.densityMesh) {
      this.densityGroup.remove(this.densityMesh);
      this.densityMesh.geometry.dispose();
      this.densityMesh.material.dispose();
      this.densityMesh = null;
    }
    const {values,shape,cell} = this.densityData;
    const res = MarchingCubes.extract(values, shape, this.isovalue, cell, this.smoothNormals);
    if (!res.positions.length) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(res.positions, 3));
    geom.setAttribute('normal',   new THREE.BufferAttribute(res.normals, 3));
    const mat = new THREE.MeshPhysicalMaterial({
      color: this.densityColor, transparent:true, opacity:this.densityOpacity,
      side:THREE.DoubleSide, roughness:0.3, metalness:0.0,
      clearcoat:0.4, depthWrite:false,
    });
    this.densityMesh = new THREE.Mesh(geom, mat);
    this.densityGroup.add(this.densityMesh);
  }
  setIsovalue(v) {
    if (!this.densityData) return;
    this.isovalue = this.densityData.min + v * (this.densityData.max - this.densityData.min);
    this.updateIsosurface();
    return this.isovalue;
  }
  setDensityOpacity(o) {
    this.densityOpacity = o;
    if (this.densityMesh) this.densityMesh.material.opacity = o;
  }
  setSmoothNormals(v) {
    this.smoothNormals = v;
    this.updateIsosurface();
  }
  setDensityColor(hex) {
    this.densityColor = new THREE.Color(hex);
    if (this.densityMesh) this.densityMesh.material.color.copy(this.densityColor);
  }

  // ═════════════════════════════════════════════════════════════════════
  //  DENSITY — Ray-marched volume renderer
  // ═════════════════════════════════════════════════════════════════════
  _getColormapData() { return Colormaps[this.currentColormap] || Colormaps.orbital; }

  _makeColormapTexture() {
    const cm = this._getColormapData();
    const d = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      d[i*4]   = Math.floor(cm[i*3]   * 255);
      d[i*4+1] = Math.floor(cm[i*3+1] * 255);
      d[i*4+2] = Math.floor(cm[i*3+2] * 255);
      d[i*4+3] = 255;
    }
    const tex = new THREE.DataTexture(d, 256, 1, THREE.RGBAFormat);
    tex.needsUpdate = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    return tex;
  }

  _buildVolumeRenderer() {
    // Clean up old
    if (this.volumeBox) {
      this.volumeGroup.remove(this.volumeBox);
      this.volumeBox.geometry.dispose();
      this.volumeBox.material.dispose();
      if (this.volumeBox.material.uniforms) {
        if (this.volumeBox.material.uniforms.uVolume.value)
          this.volumeBox.material.uniforms.uVolume.value.dispose();
        if (this.volumeBox.material.uniforms.uColormap.value)
          this.volumeBox.material.uniforms.uColormap.value.dispose();
      }
      this.volumeBox = null;
    }
    if (!this.densityData) return;

    // Check for WebGL2 / 3D texture support
    const gl = this.renderer.getContext();
    if (!gl || !(gl instanceof WebGL2RenderingContext)) {
      console.warn('WebGL2 not available; volume rendering disabled');
      return;
    }
    if (typeof THREE.DataTexture3D === 'undefined') {
      console.warn('THREE.DataTexture3D not available');
      return;
    }

    const {values, shape, cell, min, max} = this.densityData;
    const [Nx, Ny, Nz] = shape;

    // Normalize density to 0-1 for the texture
    const norm = new Float32Array(values.length);
    const range = max - min || 1;
    for (let i = 0; i < values.length; i++) {
      norm[i] = (values[i] - min) / range;
    }

    // Create 3D texture
    const volTex = new THREE.DataTexture3D(norm, Nx, Ny, Nz);
    volTex.format = THREE.RedFormat;
    volTex.type   = THREE.FloatType;
    volTex.minFilter = THREE.LinearFilter;
    volTex.magFilter = THREE.LinearFilter;
    volTex.unpackAlignment = 1;
    volTex.needsUpdate = true;

    // Colormap 1D texture
    const cmapTex = this._makeColormapTexture();

    // Box geometry [0,1]^3
    const boxGeo = new THREE.BoxBufferGeometry(1, 1, 1);
    // Shift so vertices go from (0,0,0) to (1,1,1) instead of (-0.5,-0.5,-0.5) to (0.5,0.5,0.5)
    boxGeo.translate(0.5, 0.5, 0.5);

    const mat = new THREE.RawShaderMaterial({
      uniforms: {
        uVolume:     {value: volTex},
        uColormap:   {value: cmapTex},
        uInvModel:   {value: new THREE.Matrix4()},
        uThreshold:  {value: this.volumeThresholdLow},
        uOpacity:    {value: this.volumeOpacity},
        uBrightness: {value: this.volumeBrightness},
        uSteps:      {value: this.volumeSteps},
        // Built-in uniforms provided by Three.js for RawShaderMaterial
        modelMatrix:     {value: new THREE.Matrix4()},
        modelViewMatrix: {value: new THREE.Matrix4()},
        projectionMatrix:{value: new THREE.Matrix4()},
        cameraPosition:  {value: new THREE.Vector3()},
      },
      vertexShader:   VOL_VERT,
      fragmentShader: VOL_FRAG,
      side: THREE.BackSide,  // render back faces so we always see the volume
      transparent: true,
      depthWrite: false,
    });

    this.volumeBox = new THREE.Mesh(boxGeo, mat);

    // Set model matrix = cell matrix (columns = lattice vectors)
    const a = cell[0], b = cell[1], c = cell[2];
    this.volumeBox.matrixAutoUpdate = false;
    this.volumeBox.matrix.set(
      a[0], b[0], c[0], 0,
      a[1], b[1], c[1], 0,
      a[2], b[2], c[2], 0,
      0,    0,    0,    1
    );
    this.volumeBox.matrixWorldNeedsUpdate = true;

    this.volumeGroup.add(this.volumeBox);
  }

  setVolumeThreshold(low) {
    this.volumeThresholdLow = low;
    if (this.volumeBox) this.volumeBox.material.uniforms.uThreshold.value = low;
  }
  setVolumeOpacity(o) {
    this.volumeOpacity = o;
    if (this.volumeBox) this.volumeBox.material.uniforms.uOpacity.value = o;
  }
  setVolumeBrightness(b) {
    this.volumeBrightness = b;
    if (this.volumeBox) this.volumeBox.material.uniforms.uBrightness.value = b;
  }
  setVolumeSteps(n) {
    this.volumeSteps = n;
    if (this.volumeBox) this.volumeBox.material.uniforms.uSteps.value = n;
  }
  setVolumePointSize() {} // legacy compat

  setColormap(name) {
    this.currentColormap = name;
    if (this.volumeBox) {
      const old = this.volumeBox.material.uniforms.uColormap.value;
      if (old) old.dispose();
      this.volumeBox.material.uniforms.uColormap.value = this._makeColormapTexture();
    }
    this._buildSlicePlanes();
  }

  // ═════════════════════════════════════════════════════════════════════
  //  DENSITY — Slice planes
  // ═════════════════════════════════════════════════════════════════════
  _buildSlicePlanes() {
    this.sliceGroup.clear();
    this.sliceMeshes = {x:null, y:null, z:null};
    if (!this.densityData) return;
    this._updateSlice('x', this.slicePositions.x);
    this._updateSlice('y', this.slicePositions.y);
    this._updateSlice('z', this.slicePositions.z);
  }
  _updateSlice(axis, frac) {
    if (!this.densityData) return;
    this.slicePositions[axis] = frac;
    const {values,shape,cell,min,max} = this.densityData;
    const [Nx,Ny,Nz] = shape;
    const range = max - min || 1;
    const cm = this._getColormapData();
    let W, H, si;
    if (axis==='x')      { si=Math.floor(frac*(Nx-1)); W=Ny; H=Nz; }
    else if (axis==='y') { si=Math.floor(frac*(Ny-1)); W=Nx; H=Nz; }
    else                 { si=Math.floor(frac*(Nz-1)); W=Nx; H=Ny; }

    const td = new Uint8Array(W*H*4);
    for (let j=0;j<H;j++) for (let i=0;i<W;i++) {
      let v;
      if (axis==='x')      v = values[si + Nx*(i + Ny*j)];
      else if (axis==='y') v = values[i  + Nx*(si + Ny*j)];
      else                 v = values[i  + Nx*(j + Ny*si)];
      const t = (v - min) / range;
      const rgb = Colormaps.sample(cm, t);
      const p = (j*W+i)*4;
      td[p]=Math.floor(rgb[0]*255); td[p+1]=Math.floor(rgb[1]*255);
      td[p+2]=Math.floor(rgb[2]*255); td[p+3]=220;
    }
    const tex = new THREE.DataTexture(td,W,H,THREE.RGBAFormat);
    tex.needsUpdate=true; tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;

    if (this.sliceMeshes[axis]) {
      this.sliceGroup.remove(this.sliceMeshes[axis]);
      this.sliceMeshes[axis].geometry.dispose();
      this.sliceMeshes[axis].material.dispose();
    }
    const a=cell[0], b=cell[1], c=cell[2];
    let corners;
    if (axis==='x') {
      const o=[a[0]*frac,a[1]*frac,a[2]*frac];
      corners=[ o, [o[0]+b[0],o[1]+b[1],o[2]+b[2]],
        [o[0]+b[0]+c[0],o[1]+b[1]+c[1],o[2]+b[2]+c[2]], [o[0]+c[0],o[1]+c[1],o[2]+c[2]] ];
    } else if (axis==='y') {
      const o=[b[0]*frac,b[1]*frac,b[2]*frac];
      corners=[ o, [o[0]+a[0],o[1]+a[1],o[2]+a[2]],
        [o[0]+a[0]+c[0],o[1]+a[1]+c[1],o[2]+a[2]+c[2]], [o[0]+c[0],o[1]+c[1],o[2]+c[2]] ];
    } else {
      const o=[c[0]*frac,c[1]*frac,c[2]*frac];
      corners=[ o, [o[0]+a[0],o[1]+a[1],o[2]+a[2]],
        [o[0]+a[0]+b[0],o[1]+a[1]+b[1],o[2]+a[2]+b[2]], [o[0]+b[0],o[1]+b[1],o[2]+b[2]] ];
    }
    const vts = new Float32Array([
      ...corners[0],...corners[1],...corners[2],
      ...corners[0],...corners[2],...corners[3]]);
    const uvs = new Float32Array([0,0,1,0,1,1,0,0,1,1,0,1]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(vts,3));
    geom.setAttribute('uv', new THREE.BufferAttribute(uvs,2));
    const mat = new THREE.MeshBasicMaterial({map:tex,transparent:true,side:THREE.DoubleSide,depthWrite:false});
    const mesh = new THREE.Mesh(geom, mat);
    this.sliceMeshes[axis] = mesh;
    this.sliceGroup.add(mesh);
  }
  setSlicePosition(axis, f) { this._updateSlice(axis, Math.max(0, Math.min(1, f))); }

  // ═════════════════════════════════════════════════════════════════════
  //  VISIBILITY
  // ═════════════════════════════════════════════════════════════════════
  setAtomsVisible(v)   { this.atomGroup.visible = v; }
  setBondsVisible(v)   { this.bondGroup.visible = v; }
  setCellVisible(v)    { this.cellGroup.visible = v; }
  setDensityVisible(v) { this.densityGroup.visible = v; }
  setVolumeVisible(v)  { this.volumeGroup.visible = v; }
  setSlicesVisible(v)  { this.sliceGroup.visible = v; }
  setLabelsVisible(v)  { this.labelGroup.visible = v; }

  // ═════════════════════════════════════════════════════════════════════
  //  FORCE ARROWS
  // ═════════════════════════════════════════════════════════════════════
  showForces(forces) {
    const rm = [];
    this.atomGroup.children.forEach(ch => { if (ch.userData.isForceArrow) rm.push(ch); });
    rm.forEach(c => this.atomGroup.remove(c));
    if (!forces || !forces.length) return;
    for (let i=0; i<Math.min(forces.length,this.atoms.length); i++) {
      const f = forces[i];
      const mag = Math.sqrt(f.fx**2+f.fy**2+f.fz**2);
      if (mag<1e-8) continue;
      const ar = new THREE.ArrowHelper(
        new THREE.Vector3(f.fx,f.fy,f.fz).normalize(),
        new THREE.Vector3(...this.atoms[i].position),
        Math.max(mag*30,0.3), 0xff4444, 0.3, 0.15);
      ar.userData.isForceArrow = true;
      this.atomGroup.add(ar);
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  //  COLORBAR
  // ═════════════════════════════════════════════════════════════════════
  renderColorbar(w, h) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    const cm = this._getColormapData();
    for (let x = 0; x < w; x++) {
      const rgb = Colormaps.sample(cm, x/(w-1));
      ctx.fillStyle = `rgb(${Math.floor(rgb[0]*255)},${Math.floor(rgb[1]*255)},${Math.floor(rgb[2]*255)})`;
      ctx.fillRect(x,0,1,h);
    }
    return cv;
  }

  // ═════════════════════════════════════════════════════════════════════
  //  CAMERA CONTROLS (for viewpoint panel)
  // ═════════════════════════════════════════════════════════════════════
  setFOV(val) {
    this.camera.fov = val;
    this.camera.updateProjectionMatrix();
  }

  getCameraDistance() {
    return this.camera.position.distanceTo(this.controls.target);
  }

  setCameraDistance(d) {
    const dir = this.camera.position.clone().sub(this.controls.target);
    const len = dir.length();
    if (len < 0.001) return;
    dir.normalize();
    this.camera.position.copy(this.controls.target).addScaledVector(dir, d);
    this.controls.update();
  }

  setAutoRotate(enabled, speed) {
    this.controls.autoRotate = enabled;
    this.controls.autoRotateSpeed = speed || 2.0;
  }

  resetView() {
    this._centerCamera();
  }

  // ═════════════════════════════════════════════════════════════════════
  //  GRID FLOOR (spatial reference plane)
  // ═════════════════════════════════════════════════════════════════════
  _buildGridFloor() {
    if (this._gridFloor) {
      this.scene.remove(this._gridFloor);
    }
    if (!this.cell) return;

    const size = this._cellRadius * 3;
    const divisions = 20;
    const grid = new THREE.GridHelper(size, divisions, 0x1a2030, 0x0e1219);
    grid.position.copy(this._cellCenter);
    grid.position.y = -0.5;
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
    this._gridFloor = grid;
    this.scene.add(grid);
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    this.renderer.dispose();
    this.controls.dispose();
  }
}
