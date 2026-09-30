// Thapar CFD Platform - Workflow shell: progressive disclosure (Basic/Advanced/
// Expert), the 8-stage rail (Geometry -> Physics -> Materials -> Boundaries ->
// Mesh -> Validation -> Run -> Results), case status computation, and the
// convergence monitor. Live CFD is opt-in: nothing here calls /api/sim/start
// on its own -- see onRunClicked, which only fires from the footer button.
(function() {
    const STAGES = ['geometry', 'physics', 'materials', 'boundaries', 'mesh', 'validation', 'run', 'results'];
    let currentStage = 'geometry';
    let currentLevel = 'basic';

    const MATERIALS = [
        { label: 'Air (20°C)', rho: 1.225, nu: 1.5e-5 },
        { label: 'Water (20°C)', rho: 998.2, nu: 1.0e-6 },
        { label: 'Glycerin', rho: 1260, nu: 1.19e-3 },
        { label: 'Custom…', rho: null, nu: null },
    ];

    // ---------------------------------------------------------------
    // Level (progressive disclosure) & stage navigation
    // ---------------------------------------------------------------
    window.setLevel = function(level) {
        // Dropping to a level that hides the sketch tool while a sketch is in
        // progress would leave clicks silently adding points behind a hidden
        // panel -- cancel it rather than leave that trap.
        if (level === 'basic' && window.cancelSketch) window.cancelSketch();
        currentLevel = level;
        document.body.dataset.level = level;
        document.querySelectorAll('.level-btn').forEach(b => b.classList.toggle('active', b.dataset.level === level));
        try { localStorage.setItem('thapar-level', level); } catch (e) {}
        if (level === 'expert' && currentStage === 'validation') window.loadExpertJson();
    };

    // ---------------------------------------------------------------
    // Expert stage: direct sim_spec.json IR editing (bypasses guided forms)
    // ---------------------------------------------------------------
    window.loadExpertJson = function() {
        fetch('/api/sim/config-json').then(r => r.json()).then(data => {
            document.getElementById('expert-json-editor').value = JSON.stringify(data, null, 2);
            document.getElementById('expert-json-status').textContent = 'Loaded live sim_spec.json from solver.';
        });
    };

    window.applyExpertJson = function() {
        const status = document.getElementById('expert-json-status');
        const raw = document.getElementById('expert-json-editor').value;
        try { JSON.parse(raw); } catch (e) {
            status.textContent = 'Invalid JSON: ' + e.message;
            status.style.color = 'var(--status-error)';
            return;
        }
        status.textContent = 'Applying IR to native UCOF solver…';
        status.style.color = 'var(--text-muted)';
        fetch('/api/sim/config-json', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: raw })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => {
                if (ok) {
                    status.textContent = 'Solver rebuilt from IR. Mesh must be regenerated (geometry state is unaffected but resolution/BCs may have changed).';
                    status.style.color = 'var(--status-ok)';
                    pollCaseStatus();
                    renderValidationChecklist();
                } else {
                    status.textContent = data.error || 'Apply failed';
                    status.style.color = 'var(--status-error)';
                }
            })
            .catch(err => { status.textContent = 'Apply failed: ' + err; status.style.color = 'var(--status-error)'; });
    };

    // ---------------------------------------------------------------
    // Floating/dockable panels: a small panel-manager, the same pattern behind
    // BC / Field Stats / Convergence / Probe History / Custom Equations. Each
    // panel is `.floating-panel` with a `.floating-panel-header` (drag handle
    // + collapse + close buttons) and a `.floating-panel-body`. Three distinct
    // states, deliberately not conflated:
    //   - dragged: repositioned anywhere in its viewport (startDragFloatingPanel)
    //   - collapsed: body hidden, header stays put -- "collapsible without
    //     going away" (toggleFloatingPanelCollapse)
    //   - closed: panel hidden entirely, but reachable again via a chip in the
    //     dock tray (closeFloatingPanel / openFloatingPanel), never truly lost
    // ---------------------------------------------------------------
    window.startDragFloatingPanel = function(evt, panelId) {
        if (evt.target.closest('button, select, input')) return; // don't hijack clicks on real controls
        const panel = document.getElementById(panelId);
        if (!panel) return;
        const rect = panel.getBoundingClientRect();
        const parentRect = panel.offsetParent.getBoundingClientRect();
        panel.style.position = 'absolute';
        panel.style.top = (rect.top - parentRect.top) + 'px';
        panel.style.left = (rect.left - parentRect.left) + 'px';
        panel.style.right = 'auto';
        panel.style.bottom = 'auto';
        const startX = evt.clientX, startY = evt.clientY;
        const startTop = rect.top - parentRect.top, startLeft = rect.left - parentRect.left;

        function onMove(e) {
            panel.style.left = (startLeft + (e.clientX - startX)) + 'px';
            panel.style.top = (startTop + (e.clientY - startY)) + 'px';
        }
        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        evt.preventDefault();
    };

    window.toggleFloatingPanelCollapse = function(panelId) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        const collapsed = panel.dataset.collapsed === 'true';
        panel.dataset.collapsed = collapsed ? 'false' : 'true';
        const btn = panel.querySelector('.fp-collapse-btn');
        if (btn) btn.textContent = collapsed ? '−' : '□';
        if (!collapsed) return;
        // Re-expanded: charts inside skipped drawing entirely while their canvas
        // had no offsetParent (collapsed = display:none), so give them one redraw.
        if (panelId === 'convergence-panel' && typeof drawConvergenceChart === 'function') drawConvergenceChart();
        if (panelId === 'probe-panel' && typeof drawProbeHistoryChart === 'function') drawProbeHistoryChart();
    };

    const closedFloatingPanels = new Set();
    const PANEL_REGISTRY = {
        'bc-panel': 'Boundary',
        'stats-panel': 'Field Stats',
        'convergence-panel': 'Convergence',
        'probe-panel': 'Probe History',
        'custom-eq-panel': 'Custom Equations',
    };

    window.closeFloatingPanel = function(panelId) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        panel.classList.add('hidden');
        closedFloatingPanels.add(panelId);
        renderPanelDockTray();
    };

    window.openFloatingPanel = function(panelId) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        panel.classList.remove('hidden');
        closedFloatingPanels.delete(panelId);
        renderPanelDockTray();
    };

    window.togglePanelVisibility = function(panelId) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        if (panel.classList.contains('hidden')) window.openFloatingPanel(panelId);
        else window.closeFloatingPanel(panelId);
    };

    function renderPanelDockTray() {
        const tray = document.getElementById('panel-dock-tray');
        if (!tray) return;
        const chips = Object.entries(PANEL_REGISTRY).filter(([id]) => closedFloatingPanels.has(id));
        tray.innerHTML = chips.map(([id, label]) =>
            `<button class="panel-dock-chip" onclick="window.openFloatingPanel('${id}')">+ ${label}</button>`).join('');
    }

    let stagePanelCollapsed = false;
    window.toggleStagePanel = function() {
        stagePanelCollapsed = !stagePanelCollapsed;
        const panel = document.getElementById('stage-panel');
        const btn = document.getElementById('stage-panel-collapse-btn');
        panel.style.maxHeight = stagePanelCollapsed ? '0px' : '42vh';
        panel.style.overflow = stagePanelCollapsed ? 'hidden' : 'auto';
        btn.textContent = stagePanelCollapsed ? '▼' : '▲';
    };

    window.setStage = function(stage) {
        if (stagePanelCollapsed) window.toggleStagePanel(); // switching stages should reveal the panel again
        currentStage = stage;
        document.querySelectorAll('.stage-btn').forEach(b => b.classList.toggle('active', b.dataset.stage === stage));
        document.querySelectorAll('.stage-content').forEach(el => el.classList.toggle('hidden', el.id !== 'panel-' + stage));
        if (stage === 'physics') loadExpertPhysicsFields();
        if (stage === 'boundaries') renderBoundaryList();
        if (stage === 'materials') renderMaterialButtons();
        if (stage === 'mesh') loadMeshStage();
        if (stage === 'validation') { renderValidationChecklist(); if (currentLevel === 'expert') window.loadExpertJson(); }
        if (stage === 'results') {
            fetch('/api/sim/diagnostics.json').then(r => r.json()).then(updateResultsSummary);
            loadCheckpointList();
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        let savedLevel = 'basic';
        try { savedLevel = localStorage.getItem('thapar-level') || 'basic'; } catch (e) {}
        window.setLevel(savedLevel);
        window.setStage('geometry');
        renderMaterialButtons();
        pollCaseStatus();
        setInterval(pollCaseStatus, 1200);
    });

    // ---------------------------------------------------------------
    // Physics stage
    // ---------------------------------------------------------------
    window.onPhysicsChange = function() {
        const re = parseFloat(document.getElementById('physics-re').value);
        document.getElementById('physics-re-label').textContent = re;
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            const nu = cfg.nu;
            const ub = re * nu / 1.0; // characteristic length = domain unit, matches Materials-stage nu
            document.getElementById('physics-re-flow-hint').textContent = `→ velocity scale ${ub.toFixed(4)} m/s`;
            const params = new URLSearchParams({ ub });
            const heat = document.getElementById('physics-enable-heat');
            const turb = document.getElementById('physics-enable-turb');
            const vof = document.getElementById('physics-enable-vof');
            if (heat) params.set('enableHeat', heat.checked);
            if (turb) params.set('enableTurbulence', turb.checked);
            if (vof) params.set('enableVof', vof.checked);
            fetch('/api/sim/physics', { method: 'POST', body: params }).then(() => pollCaseStatus());
        });
    };

    function loadExpertPhysicsFields() {
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
            set('expert-thermal-diffusivity', cfg.thermal_diffusivity);
            set('expert-beta-thermal', cfg.beta_thermal);
            set('expert-t-ref', cfg.t_ref);
            set('expert-gx', cfg.gx); set('expert-gy', cfg.gy); set('expert-gz', cfg.gz);
            set('expert-rho1', cfg.rho1); set('expert-rho2', cfg.rho2);
            set('expert-nu1', cfg.nu1); set('expert-nu2', cfg.nu2);
        });
    }

    window.applyExpertPhysics = function() {
        const val = (id) => parseFloat(document.getElementById(id).value);
        const params = new URLSearchParams({
            thermalDiffusivity: val('expert-thermal-diffusivity'), betaThermal: val('expert-beta-thermal'), tRef: val('expert-t-ref'),
            gx: val('expert-gx'), gy: val('expert-gy'), gz: val('expert-gz'),
            rho1: val('expert-rho1'), rho2: val('expert-rho2'), nu1: val('expert-nu1'), nu2: val('expert-nu2'),
        });
        const status = document.getElementById('expert-physics-status');
        fetch('/api/sim/physics-advanced', { method: 'POST', body: params }).then(r => r.json()).then(() => {
            if (status) { status.textContent = 'Applied.'; status.style.color = 'var(--status-ok)'; }
            pollCaseStatus();
        });
    };

    // ---------------------------------------------------------------
    // Materials stage
    // ---------------------------------------------------------------
    function renderMaterialButtons() {
        const el = document.getElementById('material-preset-buttons');
        if (!el) return;
        el.innerHTML = MATERIALS.map((m, i) => `<button class="btn text-xs" onclick="window.selectMaterialPreset(${i})">${m.label}</button>`).join('');
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            document.getElementById('material-rho').value = cfg.rho;
            document.getElementById('material-nu').value = cfg.nu;
        });
    }

    window.selectMaterialPreset = function(i) {
        const m = MATERIALS[i];
        if (m.rho === null) {
            // Custom entry needs the raw rho/nu fields, which only exist at
            // Advanced+ -- promote the level instead of silently doing nothing.
            if (currentLevel === 'basic') window.setLevel('advanced');
            document.getElementById('material-rho').focus();
            return;
        }
        document.getElementById('material-rho').value = m.rho;
        document.getElementById('material-nu').value = m.nu;
        window.applyMaterial();
    };

    window.applyMaterial = function() {
        const rho = parseFloat(document.getElementById('material-rho').value);
        const nu = parseFloat(document.getElementById('material-nu').value);
        fetch('/api/sim/material', { method: 'POST', body: new URLSearchParams({ rho, nu }) }).then(() => pollCaseStatus());
    };

    // ---------------------------------------------------------------
    // Boundaries stage (list view; the CAD viewport's floating panel is the
    // other, spatial way to do the same thing -- both write the same endpoint)
    // ---------------------------------------------------------------
    const FACE_LABELS = { xmin: 'X-Min', xmax: 'X-Max', ymin: 'Y-Min', ymax: 'Y-Max', zmin: 'Z-Min', zmax: 'Z-Max (Top)' };
    const BC_TYPES = ['no_slip_wall', 'free_slip_symmetry', 'dirichlet_inflow', 'neumann_outflow', 'moving_lid', 'periodic', 'isothermal_wall'];
    const BC_TYPE_LABELS = { no_slip_wall: 'No-Slip Wall', free_slip_symmetry: 'Free-Slip Symmetry', dirichlet_inflow: 'Inlet (Dirichlet)', neumann_outflow: 'Outlet (Neumann)', moving_lid: 'Moving Lid', periodic: 'Periodic', isothermal_wall: 'Isothermal Wall' };

    function renderBoundaryList() {
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            const list = document.getElementById('boundary-list');
            list.innerHTML = Object.keys(FACE_LABELS).map(face => {
                const type = cfg['bc_' + face + '_type'];
                const u = cfg['bc_' + face + '_u'] ?? 0;
                return `
                <div class="rounded p-2" style="background: var(--bg-inset); border: 1px solid var(--border);">
                    <div class="text-[11px] font-semibold mb-1" style="color: var(--text-primary);">${FACE_LABELS[face]}</div>
                    <select class="select-input text-[11px] w-full mb-1" id="blist-type-${face}" onchange="window.onBoundaryListTypeChange('${face}')">
                        ${BC_TYPES.map(t => `<option value="${t}" ${t === type ? 'selected' : ''}>${BC_TYPE_LABELS[t]}</option>`).join('')}
                    </select>
                    <div class="flex gap-1 items-center" id="blist-vel-${face}" style="display:${(type === 'dirichlet_inflow' || type === 'moving_lid') ? 'flex' : 'none'}">
                        <input type="number" step="0.1" value="${u}" class="select-input text-[11px] flex-1" id="blist-u-${face}" placeholder="u">
                        <button class="btn text-[11px]" onclick="window.applyBoundaryListRow('${face}')">Apply</button>
                    </div>
                    <button class="btn text-[11px] w-full mt-1" id="blist-apply-plain-${face}" style="display:${(type === 'dirichlet_inflow' || type === 'moving_lid') ? 'none' : 'block'}" onclick="window.applyBoundaryListRow('${face}')">Apply</button>
                </div>`;
            }).join('');
        });
    }

    window.onBoundaryListTypeChange = function(face) {
        const type = document.getElementById('blist-type-' + face).value;
        const needsVel = (type === 'dirichlet_inflow' || type === 'moving_lid');
        document.getElementById('blist-vel-' + face).style.display = needsVel ? 'flex' : 'none';
        document.getElementById('blist-apply-plain-' + face).style.display = needsVel ? 'none' : 'block';
    };

    window.applyBoundaryListRow = function(face) {
        const type = document.getElementById('blist-type-' + face).value;
        const uEl = document.getElementById('blist-u-' + face);
        const u = uEl ? (parseFloat(uEl.value) || 0) : 0;
        fetch('/api/sim/boundary-condition', { method: 'POST', body: new URLSearchParams({ face, type, u, v: 0, w: 0 }) })
            .then(() => { pollCaseStatus(); if (window.reloadCadGeometry) {/* face colors refresh via next fetchServerBoundaryState in cad-viewport */} });
    };

    window.onBoundaryConditionChanged = function() { pollCaseStatus(); if (currentStage === 'boundaries') renderBoundaryList(); };

    // ---------------------------------------------------------------
    // Mesh stage
    // ---------------------------------------------------------------
    function loadMeshStage() {
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            document.getElementById('mesh-nx').value = cfg.nx;
            document.getElementById('mesh-ny').value = cfg.ny;
            document.getElementById('mesh-nz').value = cfg.nz;
            document.getElementById('mesh-lx').value = (cfg.nx * cfg.dx).toFixed(4);
            document.getElementById('mesh-ly').value = (cfg.ny * cfg.dy).toFixed(4);
            document.getElementById('mesh-lz').value = (cfg.nz * cfg.dz).toFixed(4);
            const nlEl = document.getElementById('mesh-nlevel'); if (nlEl) nlEl.value = cfg.nlevel;
            const mgEl = document.getElementById('mesh-mg-iterations'); if (mgEl) mgEl.value = cfg.mg_iterations;
            updateMeshStats(cfg);
        });
    }

    function updateMeshStats(cfg) {
        const cells = cfg.nx * cfg.ny * cfg.nz;
        const lx = cfg.nx * cfg.dx, ly = cfg.ny * cfg.dy, lz = cfg.nz * cfg.dz;
        document.getElementById('mesh-stat-cells').textContent = cells.toLocaleString();
        document.getElementById('mesh-stat-extent').textContent = `${lx.toFixed(3)} × ${ly.toFixed(3)} × ${lz.toFixed(3)} m`;
        document.getElementById('mesh-stat-cellsize').textContent = `${cfg.dx.toFixed(4)}, ${cfg.dy.toFixed(4)}, ${cfg.dz.toFixed(4)}`;
        const maxE = Math.max(lx, ly, lz), minE = Math.min(lx, ly, lz);
        document.getElementById('mesh-aniso-warning').classList.toggle('hidden', (maxE / Math.max(minE, 1e-9)) <= 1.5);
    }

    window.generateMesh = function() {
        const nx = parseInt(document.getElementById('mesh-nx').value);
        const ny = parseInt(document.getElementById('mesh-ny').value);
        const nz = parseInt(document.getElementById('mesh-nz').value);
        const lx = parseFloat(document.getElementById('mesh-lx').value);
        const ly = parseFloat(document.getElementById('mesh-ly').value);
        const lz = parseFloat(document.getElementById('mesh-lz').value);
        const nlEl = document.getElementById('mesh-nlevel');
        const mgEl = document.getElementById('mesh-mg-iterations');
        const body = { nx, ny, nz, lx, ly, lz };
        if (nlEl && nlEl.value) body.nlevel = parseInt(nlEl.value);
        if (mgEl && mgEl.value) body.mgIterations = parseInt(mgEl.value);
        const status = document.getElementById('mesh-generate-status');
        status.textContent = 'Updating resolution…';
        fetch('/api/mesh/settings', { method: 'POST', body: new URLSearchParams(body) })
            .then(r => r.json()).then(cfg => {
                updateMeshStats(cfg);
                status.textContent = 'Voxelizing geometry…';
                return fetch('/api/mesh/generate', { method: 'POST' });
            })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) { status.textContent = 'Failed: ' + (data.error || 'unknown error'); status.style.color = 'var(--status-error)'; return; }
                status.textContent = `Meshed: ${data.cells.toLocaleString()} cells.`;
                status.style.color = 'var(--status-ok)';
                if (window.reloadCadGeometry) window.reloadCadGeometry();
                pollCaseStatus();
            });
    };

    window.onGeometryChanged = function() { pollCaseStatus(); };

    // ---------------------------------------------------------------
    // Validation stage
    // ---------------------------------------------------------------
    function renderValidationChecklist() {
        Promise.all([
            fetch('/api/sim/config-json').then(r => r.json()),
            fetch('/api/mesh/status').then(r => r.json()),
            fetch('/api/cad/scene').then(r => r.json()),
        ]).then(([cfg, meshStatus, scene]) => {
            const cfl = cfg.ub * cfg.dt / cfg.dx;
            const rows = [
                { ok: scene.count >= 0, label: `Geometry: ${scene.count} bod${scene.count === 1 ? 'y' : 'ies'} in scene`, level: scene.count === 0 ? 'warn' : 'ok', note: scene.count === 0 ? 'Empty domain is valid (pure fluid channel) but unusual — check this is intentional.' : '' },
                { level: 'ok', label: `Materials: ρ=${cfg.rho}, ν=${cfg.nu.toExponential(2)}` },
                { level: cfl <= 1.0 ? 'ok' : 'error', label: `CFL number: ${cfl.toFixed(3)} (target ≤ 1.0)`, note: cfl > 1.0 ? 'Reduce velocity, increase resolution spacing, or reduce dt in Mesh/Physics.' : '' },
                { level: meshStatus.upToDate ? 'ok' : 'error', label: meshStatus.upToDate ? 'Mesh: up to date' : 'Mesh: NOT generated / stale', note: meshStatus.upToDate ? '' : 'Go to the Mesh stage and click "Generate Mesh" before running.' },
            ];
            const el = document.getElementById('validation-checklist');
            el.innerHTML = rows.map(r => `
                <div class="validation-row ${r.level}">
                    <span class="mark">${r.level === 'ok' ? '✓' : (r.level === 'warn' ? '!' : '✕')}</span>
                    <div><div>${r.label}</div>${r.note ? `<div style="color:var(--text-muted); font-weight:400;">${r.note}</div>` : ''}</div>
                </div>`).join('');
        });
    }

    // ---------------------------------------------------------------
    // Case status: header chips + rail status dots (polled)
    // ---------------------------------------------------------------
    function setChip(el, level, text) {
        el.className = 'chip chip-' + level;
        el.innerHTML = `<span class="dot${level === 'active' ? ' pulse' : ''}"></span>${text}`;
    }

    function setStageStatus(stage, level, symbol) {
        const el = document.getElementById('status-' + stage);
        if (!el) return;
        el.className = 'stage-status status-' + level;
        el.textContent = symbol;
    }

    let convergenceHistory = { div: [], pres: [] };
    const MAX_HISTORY = 200;

    function pollCaseStatus() {
        Promise.all([
            fetch('/api/sim/config-json').then(r => r.json()),
            fetch('/api/mesh/status').then(r => r.json()),
            fetch('/api/cad/scene').then(r => r.json()),
            fetch('/api/sim/diagnostics.json').then(r => r.json()),
        ]).then(([cfg, meshStatus, scene, diag]) => {
            const cfl = cfg.ub * cfg.dt / cfg.dx;
            const cflOk = cfl <= 1.0;

            setStageStatus('geometry', scene.count > 0 ? 'ok' : 'warn', scene.count > 0 ? '✓' : '!');
            setStageStatus('physics', 'ok', '✓');
            setStageStatus('materials', 'ok', '✓');
            setStageStatus('boundaries', 'ok', '✓');
            setStageStatus('mesh', meshStatus.upToDate ? 'ok' : 'warn', meshStatus.upToDate ? '✓' : '!');
            setStageStatus('validation', (cflOk && meshStatus.upToDate) ? 'ok' : 'error', (cflOk && meshStatus.upToDate) ? '✓' : '✕');
            setStageStatus('run', diag.running ? 'active' : (diag.currentStep > 0 ? 'ok' : 'neutral'), diag.running ? '●' : (diag.currentStep > 0 ? '✓' : '–'));
            setStageStatus('results', diag.currentStep > 0 ? 'ok' : 'neutral', diag.currentStep > 0 ? '✓' : '–');

            const validityChip = document.getElementById('case-validity-chip');
            if (!meshStatus.upToDate) setChip(validityChip, 'error', 'Blocked: mesh not generated');
            else if (!cflOk) setChip(validityChip, 'error', `Blocked: CFL ${cfl.toFixed(2)} > 1.0`);
            else setChip(validityChip, 'ok', 'Ready to run');

            const computeChip = document.getElementById('compute-state-chip');
            if (diag.running) setChip(computeChip, 'active', 'Running');
            else if (diag.currentStep > 0) setChip(computeChip, 'warn', 'Paused');
            else setChip(computeChip, 'neutral', 'Idle');

            // Run stage live stats
            const stateEl = document.getElementById('run-stat-state');
            if (stateEl) {
                stateEl.textContent = diag.running ? 'RUNNING' : (diag.runCompleted ? 'COMPLETED' : (diag.currentStep > 0 ? 'PAUSED' : 'IDLE'));
                stateEl.style.color = diag.running ? 'var(--status-active)' : (diag.runCompleted ? 'var(--status-ok)' : (diag.currentStep > 0 ? 'var(--status-warn)' : 'var(--text-muted)'));
            }
            const stepEl = document.getElementById('run-stat-step'); if (stepEl) stepEl.textContent = diag.currentStep;
            const timeEl = document.getElementById('run-stat-time'); if (timeEl) timeEl.textContent = diag.physicalTime.toFixed(4) + ' s';
            const throughputEl = document.getElementById('run-stat-throughput'); if (throughputEl) throughputEl.textContent = Math.round(diag.framesPerSecond) + ' steps/s';
            const gpuStepEl = document.getElementById('run-stat-gpustep'); if (gpuStepEl) gpuStepEl.textContent = diag.lastStepMs.toFixed(2) + ' ms';

            const targetEl = document.getElementById('run-stat-target');
            const progressRow = document.getElementById('run-progress-row');
            if (diag.targetStep >= 0) {
                if (targetEl) targetEl.textContent = `/ ${diag.targetStep} target`;
                if (progressRow) {
                    progressRow.classList.remove('hidden');
                    const startStep = diag.targetStep - (parseInt(document.getElementById('run-target-steps')?.value) || diag.targetStep);
                    const pct = Math.min(100, Math.max(0, ((diag.currentStep - startStep) / Math.max(diag.targetStep - startStep, 1)) * 100));
                    const bar = document.getElementById('run-progress-bar');
                    if (bar) bar.style.width = pct.toFixed(1) + '%';
                    const label = document.getElementById('run-progress-label');
                    if (label) label.textContent = diag.runCompleted
                        ? `Run complete: ${diag.currentStep} steps, ${diag.physicalTime.toFixed(4)} s physical time.`
                        : `${diag.currentStep} / ${diag.targetStep} (${pct.toFixed(0)}%)`;
                }
            } else if (targetEl) {
                targetEl.textContent = '';
            }
            if (diag.runCompleted) loadCheckpointList();

            if (diag.currentStep > 0) {
                convergenceHistory.div.push(diag.maxDivergence);
                convergenceHistory.pres.push(diag.pressureResidual);
                if (convergenceHistory.div.length > MAX_HISTORY) { convergenceHistory.div.shift(); convergenceHistory.pres.shift(); }
                drawConvergenceChart();
            }

            if (currentStage === 'results') updateResultsSummary(diag);
        }).catch(() => {});
    }

    function drawConvergenceChart() {
        const canvas = document.getElementById('convergence-canvas');
        if (!canvas || !canvas.offsetParent) return; // skip work while hidden
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        // Log-scale y-axis: the standard convention for CFD residual monitors
        // (Fluent, foamMonitor) -- both quantities are error-like values that
        // decay across many orders of magnitude, which a linear axis can't show.
        const FLOOR = 1e-8;
        const allVals = [...convergenceHistory.div, ...convergenceHistory.pres].filter(v => isFinite(v) && v > 0);
        if (!allVals.length) return;
        const logMax = Math.log10(Math.max(...allVals, FLOOR));
        const logMin = Math.log10(Math.max(Math.min(...allVals), FLOOR));
        const logRange = Math.max(logMax - logMin, 1e-6);

        // gridlines + decade labels
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.fillStyle = 'var(--text-muted)';
        ctx.font = '9px ui-monospace, monospace';
        const topDecade = Math.ceil(logMax), bottomDecade = Math.floor(logMin);
        for (let d = topDecade; d >= bottomDecade; d--) {
            const y = h - ((d - logMin) / logRange) * (h - 8) - 4;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.fillText('1e' + d, 4, y - 2);
        }

        [{ data: convergenceHistory.div, color: '#4d8dff' }, { data: convergenceHistory.pres, color: '#c99a3c' }].forEach(s => {
            if (s.data.length < 2) return;
            ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
            let started = false;
            s.data.forEach((v, i) => {
                if (!isFinite(v) || v <= 0) return; // gap over non-positive values rather than distorting the log axis
                const x = (i / (MAX_HISTORY - 1)) * w;
                const norm = Math.max(0, Math.min(1, (Math.log10(v) - logMin) / logRange));
                const y = h - norm * (h - 8) - 4;
                if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
            });
            ctx.stroke();
        });
        const verdict = document.getElementById('convergence-verdict');
        if (verdict && convergenceHistory.div.length > 20) {
            const recent = convergenceHistory.div.slice(-20);
            const trend = recent[recent.length - 1] - recent[0];
            verdict.textContent = trend < 0 ? 'Trending down (converging)' : (trend > 0 ? 'Trending up (diverging?)' : 'Flat');
            verdict.style.color = trend < 0 ? 'var(--status-ok)' : (trend > 0 ? 'var(--status-error)' : 'var(--text-muted)');
        }
    }

    // ---------------------------------------------------------------
    // Results stage
    // ---------------------------------------------------------------
    function updateResultsSummary(diag) {
        const el = document.getElementById('results-summary');
        if (!el) return;
        if (!diag) { el.textContent = ''; return; }
        el.textContent = diag.currentStep > 0
            ? `${diag.currentStep} steps computed, ${diag.physicalTime.toFixed(4)} s physical time, last GPU step ${diag.lastStepMs.toFixed(2)} ms.`
            : 'No results yet — go to Run and press Start Live Preview.';
    }

    window.onFieldChange = function() {
        const sel = document.getElementById('field-select');
        document.getElementById('legend-field-label').textContent = sel.options[sel.selectedIndex].text;
    };

    // ---------------------------------------------------------------
    // Explicit opt-in Run actions (footer "Start Live Preview" button, and
    // the Run stage's "Run" button below) -- the ONLY two places in this
    // whole app that ever start the GPU compute loop. Both require a direct
    // user click; neither fires on page load or stage navigation.
    // ---------------------------------------------------------------
    window.onRunClicked = function() {
        setTimeout(pollCaseStatus, 200);
    };

    /** The "actual run" mode: N more steps, periodic checkpoints, auto-stop --
     *  distinct from Live Preview's unbounded manually-stopped stream. */
    window.startBoundedRun = function() {
        const steps = parseInt(document.getElementById('run-target-steps').value) || 1000;
        const checkpointInterval = parseInt(document.getElementById('run-checkpoint-interval').value) || 0;
        fetch('/api/sim/run', { method: 'POST', body: new URLSearchParams({ steps, checkpointInterval }) })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) { alert(data.error || 'Run failed to start'); return; }
                document.getElementById('run-progress-row').classList.remove('hidden');
                pollCaseStatus();
            });
    };

    // ---------------------------------------------------------------
    // Checkpoint browsing (Results stage): view a saved past state instead
    // of only ever watching the live GPU stream.
    // ---------------------------------------------------------------
    let checkpointCache = [];

    function loadCheckpointList() {
        fetch('/api/sim/checkpoints').then(r => r.json()).then(list => {
            checkpointCache = list;
            const sel = document.getElementById('checkpoint-select');
            const statEl = document.getElementById('run-stat-checkpoints');
            if (statEl) statEl.textContent = list.length;
            if (!sel) return;
            if (!list.length) {
                sel.innerHTML = '<option value="">No checkpoints yet</option>';
                return;
            }
            sel.innerHTML = list.map(c => `<option value="${c.step}">Step ${c.step} — t=${c.physicalTime.toFixed(4)}s</option>`).join('');
        });
    }

    window.viewCheckpoint = function() {
        const sel = document.getElementById('checkpoint-select');
        const step = parseInt(sel.value);
        if (!step && step !== 0) { alert('No checkpoint selected'); return; }
        if (window.setResultsViewMode) window.setResultsViewMode('checkpoint', step);
        const badge = document.getElementById('results-view-badge');
        badge.className = 'chip chip-warn';
        badge.innerHTML = `<span class="dot"></span>Checkpoint step ${step}`;
    };

    window.returnToLive = function() {
        if (window.setResultsViewMode) window.setResultsViewMode('live');
        const badge = document.getElementById('results-view-badge');
        badge.className = 'chip chip-active';
        badge.innerHTML = '<span class="dot pulse"></span>Live';
    };

    // ---------------------------------------------------------------
    // Point-probe pin history: press "P" while hovering the Results
    // viewport to pin the last-probed point, then poll it over time.
    // ---------------------------------------------------------------
    let pinnedProbe = null; // {x,y,z}
    let probeHistory = [];
    const MAX_PROBE_HISTORY = 200;

    window.setPinnedProbe = function(x, y, z) {
        pinnedProbe = { x, y, z };
        probeHistory = [];
        const el = document.getElementById('probe-pin-status');
        if (el) el.textContent = `Pinned (${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)})`;
        window.openFloatingPanel('probe-panel'); // pinning a point is the signal the user wants to see its history
    };

    function pollPinnedProbe() {
        if (!pinnedProbe) return;
        fetch(`/api/sim/probe?x=${pinnedProbe.x}&y=${pinnedProbe.y}&z=${pinnedProbe.z}`)
            .then(r => r.json()).then(data => {
                const fieldSelect = document.getElementById('field-select');
                const active = fieldSelect ? fieldSelect.value : 'VELOCITY';
                const key = { VELOCITY: 'speed', PRESSURE: 'pressure', TEMPERATURE: 'temperature', TURBULENCE: 'nu_t', VOF_ALPHA: 'alpha' }[active];
                const val = (key && data[key] !== undefined) ? data[key] : null;
                if (val === null) return;
                probeHistory.push(val);
                if (probeHistory.length > MAX_PROBE_HISTORY) probeHistory.shift();
                drawProbeHistoryChart();
            }).catch(() => {});
    }

    function drawProbeHistoryChart() {
        const canvas = document.getElementById('probe-history-canvas');
        if (!canvas || !canvas.offsetParent || probeHistory.length < 2) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const minV = Math.min(...probeHistory), maxV = Math.max(...probeHistory);
        const range = Math.max(maxV - minV, 1e-9);
        ctx.strokeStyle = '#4d8dff'; ctx.lineWidth = 1.5; ctx.beginPath();
        probeHistory.forEach((v, i) => {
            const x = (i / (MAX_PROBE_HISTORY - 1)) * w;
            const y = h - ((v - minV) / range) * (h - 8) - 4;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = 'var(--text-muted)'; ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(maxV.toFixed(4), 4, 10);
        ctx.fillText(minV.toFixed(4), 4, h - 4);
    }

    // ---------------------------------------------------------------
    // Custom Equations (Expert, Mesh stage): a spatial expression sigma(x,y,z)
    // evaluated over the mesh grid and merged into the same Brinkman/porosity
    // field the real CAD geometry uses -- so a custom porous baffle, graded
    // resistance zone, or patterned obstruction reaches the live GPU solver
    // through the exact channel real geometry does. This is NOT a route to an
    // arbitrary new PDE term (that needs native CUDA kernel changes); it's a
    // genuinely custom spatially-varying source/porosity field, which is the
    // feasible slice of "custom equations" without touching compiled solver code.
    // ---------------------------------------------------------------
    window.previewCustomEquation = function() {
        const expr = document.getElementById('custom-eq-input').value.trim();
        const status = document.getElementById('custom-eq-status');
        if (!expr) { status.textContent = 'Enter an expression first.'; status.style.color = 'var(--status-warn)'; return; }
        status.textContent = 'Evaluating over the mesh grid…';
        status.style.color = 'var(--text-muted)';
        fetch('/api/mesh/custom-equation/preview', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ expression: expr }),
        }).then(r => r.json().then(data => ({ ok: r.ok, data })))
          .then(({ ok, data }) => {
              if (!ok) { status.textContent = data.error || 'Invalid expression'; status.style.color = 'var(--status-error)'; return; }
              status.textContent = `min ${data.min.toFixed(4)} · max ${data.max.toFixed(4)} · mean ${data.mean.toFixed(4)} · ${data.nonzeroCells}/${data.totalCells} cells nonzero`;
              status.style.color = 'var(--status-ok)';
          }).catch(err => { status.textContent = 'Preview failed: ' + err; status.style.color = 'var(--status-error)'; });
    };

    window.applyCustomEquation = function() {
        const expr = document.getElementById('custom-eq-input').value.trim();
        const mode = document.getElementById('custom-eq-mode').value;
        const status = document.getElementById('custom-eq-status');
        if (!expr) { status.textContent = 'Enter an expression first.'; status.style.color = 'var(--status-warn)'; return; }
        status.textContent = 'Uploading custom field to GPU solver…';
        status.style.color = 'var(--text-muted)';
        fetch('/api/mesh/custom-equation/apply', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ expression: expr, mode }),
        }).then(r => r.json().then(data => ({ ok: r.ok, data })))
          .then(({ ok, data }) => {
              if (!ok) { status.textContent = data.error || 'Apply failed'; status.style.color = 'var(--status-error)'; return; }
              status.textContent = `Applied to live solver (mode: ${mode}). Takes effect on the next step.`;
              status.style.color = 'var(--status-ok)';
          }).catch(err => { status.textContent = 'Apply failed: ' + err; status.style.color = 'var(--status-error)'; });
    };

    document.addEventListener('DOMContentLoaded', () => {
        document.body.addEventListener('htmx:afterRequest', (evt) => {
            const path = evt.detail && evt.detail.pathInfo && evt.detail.pathInfo.requestPath;
            if (path && (path.includes('/api/sim/start') || path.includes('/api/sim/pause') || path.includes('/api/sim/reset') || path.includes('/api/sim/step'))) {
                pollCaseStatus();
                if (path.includes('/api/sim/reset')) { checkpointCache = []; probeHistory = []; pinnedProbe = null; }
            }
        });
        const fieldSelect = document.getElementById('field-select');
        if (fieldSelect) window.onFieldChange();
        setInterval(pollPinnedProbe, 1000);

        // Panels that start closed (probe/custom-eq are opt-in, opened on demand)
        // so the dock tray reflects reality on first load rather than the panels
        // just being invisible with no way back in.
        ['probe-panel', 'custom-eq-panel'].forEach(id => {
            const panel = document.getElementById(id);
            if (panel && panel.classList.contains('hidden')) closedFloatingPanels.add(id);
        });
        renderPanelDockTray();
    });
})();
