/**
 * LYNX DFT Web UI — Main application logic.
 * Handles config form, API calls, results display, SCF chart.
 */

(function() {
  'use strict';

  // =========================================================================
  // State
  // =========================================================================

  let viewer = null;
  let socket = null;
  let currentJobId = null;
  let scfHistory = [];
  let elementData = {};

  // =========================================================================
  // Init
  // =========================================================================

  document.addEventListener('DOMContentLoaded', async () => {
    // Initialize 3D viewer
    viewer = new LYNXViewer('viewer-3d');

    // Fetch element data
    try {
      const resp = await fetch('/api/elements');
      elementData = await resp.json();
      viewer.setElementData(elementData);
    } catch(e) {
      log('Warning: Could not load element data');
    }

    // Setup WebSocket
    setupSocket();

    // Setup UI event listeners
    setupUI();

    // Load default example
    loadExample('H2O');

    log('LYNX DFT Web UI initialized.');
  });

  // =========================================================================
  // WebSocket
  // =========================================================================

  function setupSocket() {
    socket = io();

    socket.on('connect', () => {
      log('WebSocket connected.');
    });

    socket.on('job_status', (data) => {
      if (data.job_id !== currentJobId) return;
      updateJobStatus(data.status, data.error, data.demo);
      if (data.status === 'completed') {
        loadResults(data.job_id);
      }
    });

    socket.on('scf_progress', (data) => {
      if (data.job_id !== currentJobId) return;
      scfHistory.push(data);
      drawSCFChart();
      updateSCFInfo(data);
    });

    socket.on('job_log', (data) => {
      if (data.job_id !== currentJobId) return;
      appendStdoutLine(data.line);
    });
  }

  // =========================================================================
  // UI Setup
  // =========================================================================

  function setupUI() {
    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
        // Redraw SCF chart when switching to SCF tab (canvas may have been hidden)
        if (tab.dataset.tab === 'tab-scf') {
          requestAnimationFrame(() => drawSCFChart());
        }
      });
    });

    // Panel collapse
    document.getElementById('btn-collapse-left').addEventListener('click', () => {
      document.getElementById('panel-left').classList.toggle('collapsed');
      setTimeout(() => viewer._onResize(), 250);
    });
    document.getElementById('btn-collapse-right').addEventListener('click', () => {
      document.getElementById('panel-right').classList.toggle('collapsed');
      setTimeout(() => viewer._onResize(), 250);
    });

    // Examples
    document.getElementById('btn-load-example').addEventListener('click', showExampleModal);
    document.querySelector('.modal-close').addEventListener('click', hideExampleModal);
    document.getElementById('modal-examples').addEventListener('click', (e) => {
      if (e.target.id === 'modal-examples') hideExampleModal();
    });

    // Import / Export
    document.getElementById('btn-import').addEventListener('click', () => {
      document.getElementById('file-input').click();
    });
    document.getElementById('file-input').addEventListener('change', importJSON);
    document.getElementById('btn-export').addEventListener('click', exportJSON);

    // Run / Preview / Stop
    document.getElementById('btn-run').addEventListener('click', runSimulation);
    document.getElementById('btn-stop').addEventListener('click', stopSimulation);
    document.getElementById('btn-preview').addEventListener('click', previewDensity);

    // Atom management
    document.getElementById('btn-add-atom-group').addEventListener('click', () => {
      addAtomGroup('X', [[0, 0, 0]]);
    });

    // Viewer controls
    document.getElementById('show-atoms').addEventListener('change', (e) => viewer.setAtomsVisible(e.target.checked));
    document.getElementById('show-bonds').addEventListener('change', (e) => viewer.setBondsVisible(e.target.checked));
    document.getElementById('show-cell').addEventListener('change', (e) => viewer.setCellVisible(e.target.checked));
    document.getElementById('show-labels').addEventListener('change', (e) => viewer.setLabelsVisible(e.target.checked));

    // ── Camera presets ──
    document.querySelectorAll('.cam-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        viewer.setCameraView(btn.dataset.view);
      });
    });

    // ── Density mode tabs ──
    document.querySelectorAll('.density-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.density-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.density-tab-content').forEach(c => c.classList.add('hidden'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.dtab).classList.remove('hidden');
      });
    });

    // ── Isosurface controls ──
    document.getElementById('show-density').addEventListener('change', (e) => {
      viewer.setDensityVisible(e.target.checked);
    });

    document.getElementById('iso-slider').addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      const isovalue = viewer.setIsovalue(val);
      document.getElementById('iso-value-display').textContent = isovalue ? isovalue.toFixed(4) : '--';
    });

    document.getElementById('opacity-slider').addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      viewer.setDensityOpacity(val);
      document.getElementById('opacity-display').textContent = val.toFixed(2);
    });

    document.getElementById('density-color').addEventListener('input', (e) => {
      viewer.setDensityColor(e.target.value);
    });

    // ── Volume 3D controls ──
    document.getElementById('show-volume').addEventListener('change', (e) => {
      viewer.setVolumeVisible(e.target.checked);
    });

    document.getElementById('vol-colormap').addEventListener('change', (e) => {
      viewer.setColormap(e.target.value);
      updateColorbars();
    });

    document.getElementById('vol-thresh-low').addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      document.getElementById('vol-thresh-low-display').textContent = v.toFixed(2);
      viewer.setVolumeThreshold(v);
    });

    document.getElementById('vol-brightness').addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      document.getElementById('vol-brightness-display').textContent = v.toFixed(1);
      viewer.setVolumeBrightness(v);
    });

    document.getElementById('vol-steps').addEventListener('input', (e) => {
      const v = parseInt(e.target.value);
      document.getElementById('vol-steps-display').textContent = v;
      viewer.setVolumeSteps(v);
    });

    document.getElementById('vol-opacity').addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      document.getElementById('vol-opacity-display').textContent = v.toFixed(2);
      viewer.setVolumeOpacity(v);
    });

    // ── Slice controls ──
    document.getElementById('show-slices').addEventListener('change', (e) => {
      viewer.setSlicesVisible(e.target.checked);
    });

    ['x', 'y', 'z'].forEach(axis => {
      document.getElementById('slice-' + axis).addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        document.getElementById('slice-' + axis + '-display').textContent = v.toFixed(2);
        viewer.setSlicePosition(axis, v);
      });
    });
  }

  // =========================================================================
  // Config <-> Form
  // =========================================================================

  function buildConfig() {
    const latvec = [
      [pf('lv-00'), pf('lv-01'), pf('lv-02')],
      [pf('lv-10'), pf('lv-11'), pf('lv-12')],
      [pf('lv-20'), pf('lv-21'), pf('lv-22')],
    ];

    const atoms = [];
    document.querySelectorAll('.atom-group').forEach(group => {
      const elem = group.querySelector('.atom-element').value.trim();
      const coords = [];
      group.querySelectorAll('.coord-row').forEach(row => {
        const inputs = row.querySelectorAll('input[type="number"]');
        coords.push([parseFloat(inputs[0].value), parseFloat(inputs[1].value), parseFloat(inputs[2].value)]);
      });
      if (elem && coords.length > 0) {
        atoms.push({
          element: elem,
          pseudo_file: elem + '.psp8',
          fractional: true,
          coordinates: coords,
        });
      }
    });

    return {
      lattice: {
        vectors: latvec,
        cell_type: document.getElementById('cell-type').value,
      },
      grid: {
        Nx: parseInt(document.getElementById('grid-nx').value),
        Ny: parseInt(document.getElementById('grid-ny').value),
        Nz: parseInt(document.getElementById('grid-nz').value),
        fd_order: parseInt(document.getElementById('fd-order').value),
        boundary_conditions: [
          document.getElementById('bc-x').value,
          document.getElementById('bc-y').value,
          document.getElementById('bc-z').value,
        ],
      },
      atoms: atoms,
      electronic: {
        xc: document.getElementById('xc-func').value,
        spin: document.getElementById('spin-type').value,
        temperature: parseFloat(document.getElementById('elec-temp').value),
        smearing: document.getElementById('smearing').value,
        Nstates: parseInt(document.getElementById('nstates').value),
      },
      kpoints: {
        grid: [parseInt(document.getElementById('kpt-x').value),
               parseInt(document.getElementById('kpt-y').value),
               parseInt(document.getElementById('kpt-z').value)],
        shift: [parseFloat(document.getElementById('kpt-sx').value),
                parseFloat(document.getElementById('kpt-sy').value),
                parseFloat(document.getElementById('kpt-sz').value)],
      },
      scf: {
        max_iter: parseInt(document.getElementById('scf-maxiter').value),
        tolerance: parseFloat(document.getElementById('scf-tol').value),
        mixing: document.getElementById('mixing-var').value,
        preconditioner: document.getElementById('mixing-precond').value,
        mixing_history: parseInt(document.getElementById('mixing-history').value),
        mixing_parameter: parseFloat(document.getElementById('mixing-param').value),
      },
      output: {
        print_forces: true,
        print_atoms: true,
      },
      device: document.getElementById('device-type').value,
      parallel: {
        nprocs: parseInt(document.getElementById('nprocs').value),
      },
    };
  }

  function loadConfig(config) {
    if (config.lattice) {
      const v = config.lattice.vectors;
      setVal('lv-00', v[0][0]); setVal('lv-01', v[0][1]); setVal('lv-02', v[0][2]);
      setVal('lv-10', v[1][0]); setVal('lv-11', v[1][1]); setVal('lv-12', v[1][2]);
      setVal('lv-20', v[2][0]); setVal('lv-21', v[2][1]); setVal('lv-22', v[2][2]);
      if (config.lattice.cell_type) setVal('cell-type', config.lattice.cell_type);
    }

    if (config.grid) {
      setVal('grid-nx', config.grid.Nx);
      setVal('grid-ny', config.grid.Ny);
      setVal('grid-nz', config.grid.Nz);
      if (config.grid.fd_order) setVal('fd-order', config.grid.fd_order);
      if (config.grid.boundary_conditions) {
        setVal('bc-x', config.grid.boundary_conditions[0]);
        setVal('bc-y', config.grid.boundary_conditions[1]);
        setVal('bc-z', config.grid.boundary_conditions[2]);
      }
    }

    // Atoms
    document.getElementById('atom-groups').innerHTML = '';
    if (config.atoms) {
      for (const ag of config.atoms) {
        addAtomGroup(ag.element, ag.coordinates);
      }
    }

    if (config.electronic) {
      setVal('xc-func', config.electronic.xc);
      setVal('spin-type', config.electronic.spin);
      setVal('elec-temp', config.electronic.temperature);
      setVal('smearing', config.electronic.smearing);
      setVal('nstates', config.electronic.Nstates);
    }

    if (config.kpoints) {
      setVal('kpt-x', config.kpoints.grid[0]);
      setVal('kpt-y', config.kpoints.grid[1]);
      setVal('kpt-z', config.kpoints.grid[2]);
      setVal('kpt-sx', config.kpoints.shift[0]);
      setVal('kpt-sy', config.kpoints.shift[1]);
      setVal('kpt-sz', config.kpoints.shift[2]);
    }

    if (config.scf) {
      setVal('scf-maxiter', config.scf.max_iter);
      setVal('scf-tol', config.scf.tolerance);
      setVal('mixing-var', config.scf.mixing);
      setVal('mixing-precond', config.scf.preconditioner);
      setVal('mixing-history', config.scf.mixing_history);
      setVal('mixing-param', config.scf.mixing_parameter);
    }

    // Update 3D viewer
    viewer.loadStructure(config);
  }

  // =========================================================================
  // Atom Group Editor
  // =========================================================================

  function addAtomGroup(element, coordinates) {
    const container = document.getElementById('atom-groups');
    const group = document.createElement('div');
    group.className = 'atom-group';

    let coordRows = '';
    for (const c of coordinates) {
      coordRows += coordRowHTML(c[0], c[1], c[2]);
    }

    group.innerHTML = `
      <div class="atom-group-header">
        <input type="text" class="atom-element" value="${element}" placeholder="Element" maxlength="3">
        <button class="btn-remove-group" title="Remove element">&times; Remove</button>
      </div>
      <div class="atom-group-body">
        <div class="coord-list">${coordRows}</div>
      </div>
      <div class="atom-group-actions">
        <button class="btn btn-small btn-add-coord">+ Add Atom</button>
      </div>
    `;

    // Event: remove group
    group.querySelector('.btn-remove-group').addEventListener('click', () => {
      group.remove();
      updateViewerFromForm();
    });

    // Event: add coordinate
    group.querySelector('.btn-add-coord').addEventListener('click', () => {
      const list = group.querySelector('.coord-list');
      list.insertAdjacentHTML('beforeend', coordRowHTML(0, 0, 0));
      attachCoordEvents(list.lastElementChild);
      updateViewerFromForm();
    });

    // Attach events to existing coord rows
    group.querySelectorAll('.coord-row').forEach(row => attachCoordEvents(row));

    // Update viewer when element changes
    group.querySelector('.atom-element').addEventListener('change', () => updateViewerFromForm());

    container.appendChild(group);
  }

  function coordRowHTML(x, y, z) {
    return `<div class="coord-row">
      <input type="number" value="${x}" step="0.01" placeholder="x">
      <input type="number" value="${y}" step="0.01" placeholder="y">
      <input type="number" value="${z}" step="0.01" placeholder="z">
      <button class="btn-remove-coord" title="Remove">&times;</button>
    </div>`;
  }

  function attachCoordEvents(row) {
    row.querySelector('.btn-remove-coord').addEventListener('click', () => {
      row.remove();
      updateViewerFromForm();
    });
    row.querySelectorAll('input').forEach(input => {
      input.addEventListener('change', () => updateViewerFromForm());
    });
  }

  function updateViewerFromForm() {
    try {
      const config = buildConfig();
      if (config.atoms.length > 0) {
        viewer.loadStructure(config);
      }
    } catch(e) { /* ignore partial input */ }
  }

  // =========================================================================
  // Examples
  // =========================================================================

  async function showExampleModal() {
    const modal = document.getElementById('modal-examples');
    const list = document.getElementById('example-list');

    try {
      const resp = await fetch('/api/examples');
      const examples = await resp.json();

      const descriptions = {
        'H2O': 'Water molecule. 3 atoms, Dirichlet BCs, GGA-PBE.',
        'Si8': '8-atom silicon diamond structure. Cubic cell, GGA-PBE.',
        'BaTiO3': 'Barium titanate perovskite. 5 atoms, cubic cell.',
        'BaTiO3_quick': 'BaTiO3 with smaller grid for quick testing.',
      };

      list.innerHTML = examples.map(name => `
        <div class="example-card" data-name="${name}">
          <h3>${name}</h3>
          <p>${descriptions[name] || 'LYNX DFT example input.'}</p>
        </div>
      `).join('');

      list.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
          loadExample(card.dataset.name);
          hideExampleModal();
        });
      });
    } catch(e) {
      list.innerHTML = '<p>Could not load examples.</p>';
    }

    modal.classList.remove('hidden');
  }

  function hideExampleModal() {
    document.getElementById('modal-examples').classList.add('hidden');
  }

  async function loadExample(name) {
    try {
      const resp = await fetch(`/api/examples/${name}`);
      const config = await resp.json();
      loadConfig(config);
      log(`Loaded example: ${name}`);
    } catch(e) {
      log(`Error loading example: ${e.message}`);
    }
  }

  // =========================================================================
  // Import / Export
  // =========================================================================

  function importJSON(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const config = JSON.parse(ev.target.result);
        loadConfig(config);
        log(`Imported: ${file.name}`);
      } catch(err) {
        log(`Import error: ${err.message}`);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function exportJSON() {
    const config = buildConfig();
    delete config.parallel; // Don't export parallel settings
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lynx_input.json';
    a.click();
    URL.revokeObjectURL(url);
    log('Exported configuration.');
  }

  // =========================================================================
  // Run Simulation
  // =========================================================================

  async function runSimulation() {
    const config = buildConfig();
    if (config.atoms.length === 0) {
      log('Error: No atoms defined.');
      return;
    }

    scfHistory = [];
    clearResults();
    log('Starting simulation...');
    updateJobStatus('submitting');

    try {
      const resp = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await resp.json();
      currentJobId = data.job_id;
      updateJobStatus('queued');
      log(`Job ${currentJobId} submitted.`);

      // Subscribe for updates
      socket.emit('subscribe_job', { job_id: currentJobId });

    } catch(e) {
      log(`Error: ${e.message}`);
      updateJobStatus('error');
    }
  }

  // =========================================================================
  // Stop Simulation
  // =========================================================================

  async function stopSimulation() {
    if (!currentJobId) return;
    try {
      const resp = await fetch(`/api/jobs/${currentJobId}/stop`, { method: 'POST' });
      const data = await resp.json();
      log(`Job ${currentJobId} stopped.`);
      updateJobStatus('stopped');
    } catch(e) {
      log(`Error stopping job: ${e.message}`);
    }
  }

  // =========================================================================
  // Preview (synthetic density)
  // =========================================================================

  async function previewDensity() {
    const config = buildConfig();
    if (config.atoms.length === 0) {
      log('Error: No atoms defined.');
      return;
    }

    log('Generating preview density...');

    try {
      const resp = await fetch('/api/preview-density', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await resp.json();
      if (data.error) {
        log(`Preview error: ${data.error}`);
        return;
      }

      viewer.loadDensity(data);
      onDensityLoaded();
      log('Preview density loaded. Switch between Isosurface / Volume 3D / Slices tabs.');
    } catch(e) {
      log(`Preview error: ${e.message}`);
    }
  }

  // =========================================================================
  // Load Results
  // =========================================================================

  async function loadResults(jobId) {
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      const data = await resp.json();

      if (data.demo) {
        log('Note: Using demo results (LYNX binary not built). Build with:');
        log('  cmake -B build -DBUILD_TESTS=ON && cmake --build build');
      }

      if (data.results) {
        displayResults(data.results);
      }

      if (data.scf_progress) {
        scfHistory = data.scf_progress;
        drawSCFChart();
        if (scfHistory.length > 0) {
          updateSCFInfo(scfHistory[scfHistory.length - 1]);
        }
      }

      // Load stdout log
      if (data.log && data.log.length > 0) {
        clearStdout();
        for (const line of data.log) {
          appendStdoutLine(line);
        }
      }

      // Load density
      const densResp = await fetch(`/api/jobs/${jobId}/density`);
      const densData = await densResp.json();
      if (!densData.error) {
        viewer.loadDensity(densData);
        onDensityLoaded();
      }

      log('Results loaded successfully.');
    } catch(e) {
      log(`Error loading results: ${e.message}`);
    }
  }

  function displayResults(results) {
    // Energy
    if (results.energy) {
      const tbody = document.querySelector('#energy-table tbody');
      tbody.innerHTML = '';
      const entries = [
        ['Total', results.energy.total],
        ['Band', results.energy.band],
        ['Exchange-Correlation', results.energy.xc],
        ['Hartree', results.energy.hartree],
        ['Self Energy', results.energy.self],
        ['Correction', results.energy.correction],
        ['Entropy (-TS)', results.energy.entropy],
        ['Per Atom', results.energy.per_atom],
      ];
      for (const [name, val] of entries) {
        const row = document.createElement('tr');
        const cls = name === 'Total' ? ' style="font-weight:600;color:var(--accent)"' : '';
        row.innerHTML = `<td${cls}>${name}</td><td${cls}>${val != null ? val.toFixed(6) : '--'}</td>`;
        tbody.appendChild(row);
      }
      document.getElementById('energy-table').classList.remove('hidden');
      document.getElementById('energy-placeholder').classList.add('hidden');
    }

    // Forces
    if (results.forces && results.forces.length > 0) {
      const tbody = document.querySelector('#forces-table tbody');
      tbody.innerHTML = '';
      for (const f of results.forces) {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${f.atom}</td><td>${f.element}</td>
          <td>${f.fx.toFixed(6)}</td><td>${f.fy.toFixed(6)}</td><td>${f.fz.toFixed(6)}</td>`;
        tbody.appendChild(row);
      }
      document.getElementById('forces-table').classList.remove('hidden');
      document.getElementById('forces-placeholder').classList.add('hidden');

      // Show force arrows
      viewer.showForces(results.forces);
    }

    // Stress
    if (results.stress) {
      const s = results.stress.voigt || [0,0,0,0,0,0];
      // Voigt order from parser: [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz]
      const fmt = v => typeof v === 'number' ? v.toFixed(4) : '--';
      document.getElementById('s-xx').textContent = fmt(s[0]);
      document.getElementById('s-yy').textContent = fmt(s[3]);
      document.getElementById('s-zz').textContent = fmt(s[5]);
      document.getElementById('s-xy').textContent = fmt(s[1]);
      document.getElementById('s-yz').textContent = fmt(s[4]);
      document.getElementById('s-xz').textContent = fmt(s[2]);
      document.getElementById('pressure-val').textContent =
        (results.stress.pressure_GPa || 0).toFixed(4) + ' GPa';
      document.getElementById('stress-display').classList.remove('hidden');
      document.getElementById('stress-placeholder').classList.add('hidden');
    }

    // SCF info
    if (results.scf) {
      const scf = results.scf;
      document.getElementById('scf-niter').textContent = scf.n_iterations || '--';
      document.getElementById('scf-converged').textContent = scf.converged ? 'Yes' : 'No';
      document.getElementById('scf-converged').className = 'info-value ' + (scf.converged ? 'text-green' : 'text-red');
      document.getElementById('scf-residual').textContent = scf.final_residual ? scf.final_residual.toExponential(2) : '--';
    }

    if (results.fermi_energy != null) {
      document.getElementById('scf-fermi').textContent = results.fermi_energy.toFixed(6) + ' Ha';
    }
  }

  function clearResults() {
    document.getElementById('energy-table').classList.add('hidden');
    document.getElementById('energy-placeholder').classList.remove('hidden');
    document.getElementById('forces-table').classList.add('hidden');
    document.getElementById('forces-placeholder').classList.remove('hidden');
    document.getElementById('stress-display').classList.add('hidden');
    document.getElementById('stress-placeholder').classList.remove('hidden');
    document.getElementById('scf-chart').classList.add('hidden');
    document.getElementById('scf-placeholder').classList.remove('hidden');
    document.getElementById('scf-info').classList.add('hidden');

    document.getElementById('scf-niter').textContent = '--';
    document.getElementById('scf-converged').textContent = '--';
    document.getElementById('scf-residual').textContent = '--';
    document.getElementById('scf-fermi').textContent = '--';

    clearStdout();
  }

  // =========================================================================
  // SCF Chart (Canvas 2D)
  // =========================================================================

  function drawSCFChart() {
    if (scfHistory.length === 0) return;

    const canvas = document.getElementById('scf-chart');
    canvas.classList.remove('hidden');
    document.getElementById('scf-placeholder').classList.add('hidden');
    document.getElementById('scf-info').classList.remove('hidden');

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    // Skip drawing if canvas is hidden (zero size) — will redraw on tab switch
    if (rect.width < 10 || rect.height < 10) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Clear
    ctx.fillStyle = '#080b10';
    ctx.fillRect(0, 0, W, H);

    const pad = { top: 20, right: 20, bottom: 35, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Data: plot log10(residual) vs iteration
    const residuals = scfHistory.map(d => Math.log10(Math.max(d.residual, 1e-15)));
    const iters = scfHistory.map(d => d.iter);

    const xMin = Math.min(...iters);
    const xMax = Math.max(...iters);
    const yMin = Math.min(...residuals) - 0.5;
    const yMax = Math.max(...residuals) + 0.5;

    function toX(i) { return pad.left + ((i - xMin) / (xMax - xMin || 1)) * plotW; }
    function toY(v) { return pad.top + ((yMax - v) / (yMax - yMin || 1)) * plotH; }

    // Grid lines
    ctx.strokeStyle = '#151a23';
    ctx.lineWidth = 1;
    const nGridY = 5;
    for (let i = 0; i <= nGridY; i++) {
      const yVal = yMin + (yMax - yMin) * i / nGridY;
      const y = toY(yVal);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#4a5568';
      ctx.font = '10px IBM Plex Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText('1e' + yVal.toFixed(0), pad.left - 5, y + 3);
    }

    // Axes
    ctx.strokeStyle = '#1e2738';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = '#4a5568';
    ctx.textAlign = 'center';
    ctx.font = '10px IBM Plex Mono, monospace';
    const step = Math.max(1, Math.ceil(iters.length / 8));
    for (let i = 0; i < iters.length; i += step) {
      ctx.fillText(iters[i], toX(iters[i]), pad.top + plotH + 15);
    }

    // Labels
    ctx.fillStyle = '#4a5568';
    ctx.font = '11px DM Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SCF Iteration', W / 2, H - 5);

    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Residual', 0, 0);
    ctx.restore();

    // Data line with glow
    ctx.shadowColor = 'rgba(59, 130, 246, 0.4)';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < iters.length; i++) {
      const x = toX(iters[i]);
      const y = toY(residuals[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Data points
    ctx.fillStyle = '#3b82f6';
    for (let i = 0; i < iters.length; i++) {
      const x = toX(iters[i]);
      const y = toY(residuals[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function updateSCFInfo(entry) {
    document.getElementById('scf-niter').textContent = entry.iter;
    if (entry.residual) {
      document.getElementById('scf-residual').textContent = entry.residual.toExponential(2);
    }
    if (entry.fermi) {
      document.getElementById('scf-fermi').textContent = entry.fermi.toFixed(6) + ' Ha';
    }
    document.getElementById('scf-info').classList.remove('hidden');
  }

  // =========================================================================
  // Job Status
  // =========================================================================

  function updateJobStatus(status, error, demo) {
    const indicator = document.getElementById('job-indicator');
    const text = document.getElementById('job-status-text');
    const pulse = indicator.querySelector('.pulse');

    indicator.classList.remove('hidden');

    const statusMap = {
      submitting: { text: 'Submitting...', color: 'var(--orange)' },
      queued: { text: 'Queued', color: 'var(--orange)' },
      running: { text: 'Running SCF...', color: 'var(--green)' },
      completed: { text: demo ? 'Completed (Demo)' : 'Completed', color: 'var(--accent)' },
      error: { text: 'Error', color: 'var(--red)' },
      stopped: { text: 'Stopped', color: 'var(--text-muted)' },
    };

    const s = statusMap[status] || { text: status, color: 'var(--text-muted)' };
    text.textContent = s.text;
    pulse.style.background = s.color;

    if (status === 'running') {
      pulse.style.animation = 'pulse-anim 1.5s infinite';
    } else {
      pulse.style.animation = 'none';
    }

    if (error) {
      log('Error: ' + error);
    }

    // Update run / stop buttons
    const btn = document.getElementById('btn-run');
    const btnStop = document.getElementById('btn-stop');
    const isRunning = (status === 'running' || status === 'queued' || status === 'submitting');
    if (isRunning) {
      btn.textContent = 'Running...';
      btn.disabled = true;
      btnStop.classList.remove('hidden');
    } else {
      btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2l10 6-10 6V2z"/></svg> Run Simulation';
      btn.disabled = false;
      btnStop.classList.add('hidden');
    }
  }

  // =========================================================================
  // Logging
  // =========================================================================

  function log(msg) {
    const el = document.getElementById('log-output');
    const time = new Date().toLocaleTimeString();
    el.textContent += `\n[${time}] ${msg}`;
    el.scrollTop = el.scrollHeight;
  }

  function appendStdoutLine(line) {
    const el = document.getElementById('stdout-output');
    if (!el) return;
    // Show the stdout section
    el.closest('.stdout-section').classList.remove('hidden');
    el.textContent += line + '\n';
    el.scrollTop = el.scrollHeight;
  }

  function clearStdout() {
    const el = document.getElementById('stdout-output');
    if (el) el.textContent = '';
  }

  // =========================================================================
  // Helpers
  // =========================================================================

  // =========================================================================
  // Colorbar rendering
  // =========================================================================

  function updateColorbars() {
    if (!viewer.densityData) return;
    const min = viewer.densityData.min;
    const max = viewer.densityData.max;

    // Volume colorbar
    const canvas1 = document.getElementById('colorbar-canvas');
    if (canvas1) {
      const bar = viewer.renderColorbar(canvas1.width, canvas1.height);
      const ctx = canvas1.getContext('2d');
      ctx.drawImage(bar, 0, 0, canvas1.width, canvas1.height);
    }
    document.getElementById('colorbar-min').textContent = min.toFixed(3);
    document.getElementById('colorbar-max').textContent = max.toFixed(3);

    // Slice colorbar
    const canvas2 = document.getElementById('colorbar-canvas-s');
    if (canvas2) {
      const bar = viewer.renderColorbar(canvas2.width, canvas2.height);
      const ctx = canvas2.getContext('2d');
      ctx.drawImage(bar, 0, 0, canvas2.width, canvas2.height);
    }
    document.getElementById('colorbar-min-s').textContent = min.toFixed(3);
    document.getElementById('colorbar-max-s').textContent = max.toFixed(3);
  }

  /**
   * Called after density data is loaded (from preview or job results).
   * Enables all density visualization modes and updates UI.
   */
  function onDensityLoaded() {
    // Enable isosurface
    document.getElementById('show-density').checked = true;
    viewer.setDensityVisible(true);
    const iso = viewer.densityData.min + 0.3 * (viewer.densityData.max - viewer.densityData.min);
    document.getElementById('iso-value-display').textContent = iso.toFixed(4);

    // Enable volume
    document.getElementById('show-volume').checked = true;
    viewer.setVolumeVisible(true);

    // Enable slices
    document.getElementById('show-slices').checked = true;
    viewer.setSlicesVisible(true);

    // Activate Volume tab by default (the new feature)
    document.querySelectorAll('.density-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.density-tab-content').forEach(c => c.classList.add('hidden'));
    document.querySelector('[data-dtab="dtab-volume"]').classList.add('active');
    document.getElementById('dtab-volume').classList.remove('hidden');

    updateColorbars();
  }

  function pf(id) { return parseFloat(document.getElementById(id).value) || 0; }
  function setVal(id, val) {
    const el = document.getElementById(id);
    if (el && val != null) el.value = val;
  }

})();
