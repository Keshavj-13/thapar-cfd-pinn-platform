// Thapar CFD Platform - Three.js CAD Viewport
// Multi-body parametric CAD scene: real OCCT primitives/boolean/fillet/sketch
// (see tools/cad_kernel_freecad.py, real FreeCAD), domain boundary-condition face-picking, a
// navigation gizmo, shaded/wireframe/x-ray/section display modes.
(function() {
    let scene, camera, renderer, controls;
    let raycaster, mouse;
    let sceneData = null;          // last /api/sim/cad-geometry response
    let bodyMeshes = {};            // bodyId -> { mesh, wireframeMesh, data }
    let selectedBodyId = null;

    // ---- Domain boundary patches (6 outer faces of the [0,1]^3 viewport cube) ----
    const FACE_DEFS = [
        { key: 'xmin', center: [0, 0.5, 0.5], rot: [0, -Math.PI / 2, 0] },
        { key: 'xmax', center: [1, 0.5, 0.5], rot: [0, Math.PI / 2, 0] },
        { key: 'ymin', center: [0.5, 0, 0.5], rot: [Math.PI / 2, 0, 0] },
        { key: 'ymax', center: [0.5, 1, 0.5], rot: [-Math.PI / 2, 0, 0] },
        { key: 'zmin', center: [0.5, 0.5, 0], rot: [Math.PI, 0, 0] },
        { key: 'zmax', center: [0.5, 0.5, 1], rot: [0, 0, 0] },
    ];
    const FACE_LABELS = { xmin: 'X-Min', xmax: 'X-Max', ymin: 'Y-Min', ymax: 'Y-Max', zmin: 'Z-Min', zmax: 'Z-Max (Top)' };
    const BC_COLORS = {
        no_slip_wall: 0x64748b, free_slip_symmetry: 0xa78bfa, dirichlet_inflow: 0x22c55e,
        neumann_outflow: 0xf97316, moving_lid: 0xf59e0b, periodic: 0x14b8a6, isothermal_wall: 0xf43f5e,
    };
    const faceState = {
        xmin: { type: 'no_slip_wall', u: 0, v: 0, w: 0 }, xmax: { type: 'no_slip_wall', u: 0, v: 0, w: 0 },
        ymin: { type: 'no_slip_wall', u: 0, v: 0, w: 0 }, ymax: { type: 'no_slip_wall', u: 0, v: 0, w: 0 },
        zmin: { type: 'no_slip_wall', u: 0, v: 0, w: 0 }, zmax: { type: 'moving_lid', u: 1, v: 0, w: 0 },
    };
    let domainFaceMeshes = {};
    let domainFacesGroup = null;
    let selectedFace = null;

    let gizmoScene, gizmoCamera;
    const GIZMO_SIZE = 84;

    let shadingMode = 'shaded';
    let sectionEnabled = false;
    let sectionPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0.5);

    // ---- Sketch tool state ----
    let sketchActive = false;
    let sketchPoints = [];
    let sketchPlane = 'xy';
    let constructionPlaneMesh = null;
    let sketchPreviewGroup = new THREE.Group();

    function init() {
        const container = document.getElementById('cad-container');
        if (!container) return;
        const width = container.clientWidth, height = container.clientHeight;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x121316);

        camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
        camera.position.set(1.6, 1.3, 1.9);

        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.localClippingEnabled = true;
        renderer.autoClear = false;
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0.5, 0.5, 0.5);

        scene.add(new THREE.AmbientLight(0xffffff, 0.65));
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight1.position.set(5, 10, 7);
        scene.add(dirLight1);
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.25);
        dirLight2.position.set(-5, -5, -5);
        scene.add(dirLight2);

        const grid = new THREE.GridHelper(2, 20, 0x262a30, 0x1b1d21);
        grid.position.set(0.5, -0.001, 0.5);
        scene.add(grid);

        scene.add(sketchPreviewGroup);

        buildDomainFaces();
        buildGizmo();

        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();

        renderer.domElement.addEventListener('pointerdown', onPointerDown, false);
        window.addEventListener('resize', onWindowResize, false);

        fetchServerBoundaryState();
        loadCadGeometry();
        renderPrimitiveForm();
        animate();
    }

    // ---------------------------------------------------------------
    // Domain boundary patches
    // ---------------------------------------------------------------
    function buildDomainFaces() {
        domainFacesGroup = new THREE.Group();
        FACE_DEFS.forEach(def => {
            const geo = new THREE.PlaneGeometry(1, 1);
            const mat = new THREE.MeshBasicMaterial({ color: BC_COLORS[faceState[def.key].type], transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(...def.center);
            mesh.rotation.set(...def.rot);
            mesh.userData = { isDomainFace: true, face: def.key };
            domainFacesGroup.add(mesh);
            domainFaceMeshes[def.key] = mesh;

            const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x2c3036 }));
            edges.position.copy(mesh.position); edges.rotation.copy(mesh.rotation);
            domainFacesGroup.add(edges);
        });
        scene.add(domainFacesGroup);
        refreshFaceColors();
    }

    function refreshFaceColors() {
        Object.entries(domainFaceMeshes).forEach(([key, mesh]) => {
            mesh.material.color.setHex(BC_COLORS[faceState[key].type]);
            mesh.material.opacity = (selectedFace === key) ? 0.5 : 0.18;
        });
    }

    function fetchServerBoundaryState() {
        fetch('/api/sim/config-json').then(r => r.json()).then(cfg => {
            ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'].forEach(face => {
                const type = cfg['bc_' + face + '_type'];
                if (type) faceState[face] = { type, u: cfg['bc_' + face + '_u'] ?? 0, v: cfg['bc_' + face + '_v'] ?? 0, w: cfg['bc_' + face + '_w'] ?? 0 };
            });
            refreshFaceColors();
            if (selectedFace) renderBcPanel(selectedFace);
        }).catch(() => {});
    }

    // ---------------------------------------------------------------
    // Navigation gizmo
    // ---------------------------------------------------------------
    function buildGizmo() {
        gizmoScene = new THREE.Scene();
        gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
        gizmoCamera.position.set(0, 0, 4);
        [{ dir: [1, 0, 0], color: 0xc85c52, label: 'X' }, { dir: [0, 1, 0], color: 0x4f9d6e, label: 'Y' }, { dir: [0, 0, 1], color: 0x4d8dff, label: 'Z' }]
            .forEach(a => {
                const dir = new THREE.Vector3(...a.dir);
                gizmoScene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 1.1, a.color, 0.26, 0.13));
                const sprite = makeTextSprite(a.label, a.color);
                sprite.position.copy(dir.clone().multiplyScalar(1.45));
                gizmoScene.add(sprite);
            });
        gizmoScene.add(new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 16), new THREE.MeshBasicMaterial({ color: 0x2c3036 })));
    }

    function makeTextSprite(text, color) {
        const canvas = document.createElement('canvas');
        canvas.width = 56; canvas.height = 56;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
        ctx.beginPath(); ctx.arc(28, 28, 22, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#121316';
        ctx.font = '600 26px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(text, 28, 29);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
        sprite.scale.set(0.38, 0.38, 1);
        return sprite;
    }

    function renderGizmo() {
        const container = document.getElementById('cad-container');
        if (!container) return;
        gizmoCamera.quaternion.copy(camera.quaternion);
        gizmoCamera.position.copy(camera.position).normalize().multiplyScalar(4);
        renderer.setViewport(0, 0, container.clientWidth, container.clientHeight);
        renderer.setScissorTest(false);
        renderer.clear();
        renderer.render(scene, camera);
        const pad = 8;
        renderer.setViewport(container.clientWidth - GIZMO_SIZE - pad, container.clientHeight - GIZMO_SIZE - pad, GIZMO_SIZE, GIZMO_SIZE);
        renderer.setScissor(container.clientWidth - GIZMO_SIZE - pad, container.clientHeight - GIZMO_SIZE - pad, GIZMO_SIZE, GIZMO_SIZE);
        renderer.setScissorTest(true);
        renderer.clearDepth();
        renderer.render(gizmoScene, gizmoCamera);
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, container.clientWidth, container.clientHeight);
    }

    // ---------------------------------------------------------------
    // Multi-body scene rendering
    // ---------------------------------------------------------------
    function loadCadGeometry() {
        fetch('/api/sim/cad-geometry').then(r => r.json()).then(data => { sceneData = data; renderScene(data); })
            .catch(err => console.warn("No CAD geometry loaded yet", err));
    }

    function renderScene(data) {
        Object.values(bodyMeshes).forEach(bm => { scene.remove(bm.mesh); scene.remove(bm.wireframeMesh); });
        bodyMeshes = {};
        let totalTris = 0;
        (data.bodies || []).forEach(body => {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(body.vertices, 3));
            if (body.normals && body.normals.length) geometry.setAttribute('normal', new THREE.Float32BufferAttribute(body.normals, 3));
            else geometry.computeVertexNormals();
            geometry.setIndex(body.indices);

            const material = new THREE.MeshStandardMaterial({ color: 0x4d8dff, roughness: 0.45, metalness: 0.35, side: THREE.DoubleSide, clippingPlanes: sectionEnabled ? [sectionPlane] : [] });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData = { bodyId: body.id };
            scene.add(mesh);

            const wireframeMesh = new THREE.LineSegments(new THREE.WireframeGeometry(geometry), new THREE.LineBasicMaterial({ color: 0x2c4a80, transparent: true, opacity: 0.3 }));
            scene.add(wireframeMesh);

            bodyMeshes[body.id] = { mesh, wireframeMesh, data: body };
            totalTris += body.num_faces;
        });
        applyShadingModeAll();
        const triEl = document.getElementById('cad-triangle-count');
        if (triEl) triEl.textContent = totalTris;
        renderBodyList(data.bodies || []);
    }

    const EDITABLE_TYPES = new Set(['box', 'cylinder', 'sphere', 'cone', 'torus', 'airfoil']);

    function renderBodyList(bodies) {
        const list = document.getElementById('body-list');
        if (!list) return;
        if (!bodies.length) {
            list.innerHTML = '<span class="text-[11px] px-2 py-1" style="color: var(--text-muted);">No bodies yet — add a primitive or import a file.</span>';
        } else {
            list.innerHTML = bodies.map(b => `
                <div class="body-list-row ${b.id === selectedBodyId ? 'selected' : ''}" onclick="window.selectBody('${b.id}')">
                    <span style="flex:1;">${b.label} <span style="color:var(--text-muted);">(${b.type}${b.watertight ? '' : ', open'})</span></span>
                    <span style="color:var(--text-muted);">${b.num_faces} tri</span>
                    ${EDITABLE_TYPES.has(b.type) ? `<button onclick="event.stopPropagation(); window.startEditBody('${b.id}')" style="color: var(--accent);" title="Edit dimensions/position">✎</button>` : ''}
                    <button onclick="event.stopPropagation(); window.deleteBody('${b.id}')" style="color: var(--status-error);" title="Delete">&times;</button>
                </div>`).join('');
        }
        ['bool-a-select', 'bool-b-select', 'fillet-body-select'].forEach(selId => {
            const sel = document.getElementById(selId);
            if (!sel) return;
            const prev = sel.value;
            sel.innerHTML = bodies.map(b => `<option value="${b.id}">${b.label}</option>`).join('');
            if (bodies.some(b => b.id === prev)) sel.value = prev;
        });
    }

    window.selectBody = function(id) {
        selectedBodyId = (selectedBodyId === id) ? null : id;
        renderBodyList(sceneData ? sceneData.bodies : []);
        highlightSelectedBody();
    };

    function highlightSelectedBody() {
        Object.entries(bodyMeshes).forEach(([id, bm]) => {
            bm.mesh.material.emissive = new THREE.Color(id === selectedBodyId ? 0x1a3a6b : 0x000000);
        });
        const badge = document.getElementById('selected-face-badge');
        if (badge && selectedBodyId && bodyMeshes[selectedBodyId]) {
            badge.textContent = bodyMeshes[selectedBodyId].data.label + ' (solid, no-slip)';
        } else if (badge) {
            badge.textContent = 'Click a face';
        }
    }

    window.deleteBody = function(id) {
        const label = (bodyMeshes[id] && bodyMeshes[id].data.label) || id;
        if (!confirm(`Delete body "${label}"? This cannot be undone.`)) return;
        fetch('/api/cad/bodies/' + id, { method: 'DELETE' }).then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) { alert(data.error || 'Cannot delete this body'); return; }
                sceneData = data; renderScene(data);
            });
    };

    function onPointerDown(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        if (sketchActive) {
            if (!constructionPlaneMesh) return;
            const hits = raycaster.intersectObject(constructionPlaneMesh);
            if (hits.length > 0) addSketchPoint(hits[0].point);
            return;
        }

        const domainHits = raycaster.intersectObjects(Object.values(domainFaceMeshes));
        if (domainHits.length > 0) {
            selectFace(domainHits[0].object.userData.face);
            return;
        }

        const bodyHits = raycaster.intersectObjects(Object.values(bodyMeshes).map(bm => bm.mesh));
        if (bodyHits.length > 0) {
            window.selectBody(bodyHits[0].object.userData.bodyId);
        }
    }

    function selectFace(faceKey) {
        selectedFace = faceKey;
        refreshFaceColors();
        renderBcPanel(faceKey);
    }

    function renderBcPanel(faceKey) {
        const panel = document.getElementById('bc-panel');
        if (!panel) return;
        panel.classList.remove('hidden');
        document.getElementById('bc-panel-face-label').textContent = FACE_LABELS[faceKey];
        const state = faceState[faceKey];
        document.getElementById('bc-type-select').value = state.type;
        document.getElementById('bc-u-input').value = state.u;
        document.getElementById('bc-v-input').value = state.v;
        document.getElementById('bc-w-input').value = state.w;
        updateBcInputVisibility(state.type);
    }

    window.updateBcInputVisibility = function(type) {
        const row = document.getElementById('bc-velocity-row');
        if (row) row.style.display = (type === 'dirichlet_inflow' || type === 'moving_lid') ? 'flex' : 'none';
    };

    window.applyBoundaryCondition = function() {
        if (!selectedFace) return;
        const type = document.getElementById('bc-type-select').value;
        const u = parseFloat(document.getElementById('bc-u-input').value) || 0;
        const v = parseFloat(document.getElementById('bc-v-input').value) || 0;
        const w = parseFloat(document.getElementById('bc-w-input').value) || 0;
        const btn = document.getElementById('bc-apply-btn');
        if (btn) { btn.disabled = true; btn.textContent = 'Applying…'; }
        fetch('/api/sim/boundary-condition', { method: 'POST', body: new URLSearchParams({ face: selectedFace, type, u, v, w }) })
            .then(r => r.json()).then(() => {
                faceState[selectedFace] = { type, u, v, w };
                refreshFaceColors();
                if (btn) { btn.disabled = false; btn.textContent = 'Apply Boundary Condition'; }
                if (window.onBoundaryConditionChanged) window.onBoundaryConditionChanged();
            }).catch(() => { if (btn) { btn.disabled = false; btn.textContent = 'Apply Boundary Condition'; } });
    };

    window.closeBcPanel = function() {
        selectedFace = null;
        refreshFaceColors();
        document.getElementById('bc-panel').classList.add('hidden');
    };

    // ---------------------------------------------------------------
    // Parametric primitives
    // ---------------------------------------------------------------
    const PRIMITIVE_FIELDS = {
        box: [['w', 0.2], ['h', 0.2], ['d', 0.2]],
        cylinder: [['radius', 0.1], ['height', 0.4]],
        sphere: [['radius', 0.1]],
        cone: [['radius1', 0.15], ['radius2', 0.0], ['height', 0.3]],
        torus: [['radius_major', 0.2], ['radius_minor', 0.05]],
        airfoil: [['chord', 0.5], ['thickness', 0.06], ['span', 0.6]],
    };

    // Inputs are pre-filled with sensible defaults, so a placeholder alone is
    // invisible from the start (placeholders only show on an empty field) --
    // every field gets a real, persistent label above it instead.
    function labeledField(labelText, inputHtml) {
        return `<div><label class="text-[10px] font-mono block mb-0.5" style="color: var(--text-muted);">${labelText}</label>${inputHtml}</div>`;
    }

    let editingBodyId = null;

    window.renderPrimitiveForm = function(prefillParams, prefillTransform) {
        const type = document.getElementById('primitive-type-select').value;
        const container = document.getElementById('primitive-form');
        const fields = PRIMITIVE_FIELDS[type] || [];
        const p = prefillParams || {};
        const t = prefillTransform || {};
        let html = fields.map(([name, def]) =>
            labeledField(name + ' (m)', `<input class="select-input text-xs w-full" data-param="${name}" type="number" step="0.01" value="${p[name] !== undefined ? p[name] : def}">`)
        ).join('');
        if (type === 'cylinder') {
            const axis = p.axis || 'z';
            html += labeledField('axis', `<select class="select-input text-xs w-full" data-param="axis">${['z', 'x', 'y'].map(a => `<option value="${a}" ${a === axis ? 'selected' : ''}>${a}</option>`).join('')}</select>`);
        }
        html += labeledField('x pos (m)', `<input class="select-input text-xs w-full" data-param="_tx" type="number" step="0.01" value="${t.tx !== undefined ? t.tx : 0.5}">`);
        html += labeledField('y pos (m)', `<input class="select-input text-xs w-full" data-param="_ty" type="number" step="0.01" value="${t.ty !== undefined ? t.ty : 0.5}">`);
        html += labeledField('z pos (m)', `<input class="select-input text-xs w-full" data-param="_tz" type="number" step="0.01" value="${t.tz !== undefined ? t.tz : 0.5}">`);
        container.innerHTML = html;
    };

    /** Loads a body's real params/transform from the server (the tessellated
     *  preview alone doesn't carry them) into the Add Primitive form and
     *  switches it into "Update" mode -- editing a shape in place instead of
     *  only ever being able to delete-and-recreate it. */
    window.startEditBody = function(id) {
        fetch('/api/cad/scene').then(r => r.json()).then(scene => {
            const body = (scene.bodies || []).find(b => b.id === id);
            if (!body) return;
            editingBodyId = id;
            document.getElementById('primitive-type-select').value = body.type;
            document.getElementById('primitive-type-select').disabled = true;
            window.renderPrimitiveForm(body.params, body.transform);
            const btn = document.getElementById('add-body-btn');
            if (btn) btn.textContent = 'Update ' + body.label;
            const cancelBtn = document.getElementById('cancel-edit-btn');
            if (cancelBtn) cancelBtn.classList.remove('hidden');
        });
    };

    window.cancelEditBody = function() {
        editingBodyId = null;
        document.getElementById('primitive-type-select').disabled = false;
        window.renderPrimitiveForm();
        const btn = document.getElementById('add-body-btn');
        if (btn) btn.textContent = 'Add Body';
        const cancelBtn = document.getElementById('cancel-edit-btn');
        if (cancelBtn) cancelBtn.classList.add('hidden');
    };

    window.addPrimitive = function() {
        const type = document.getElementById('primitive-type-select').value;
        const inputs = document.querySelectorAll('#primitive-form [data-param]');
        const params = {}, transform = {};
        inputs.forEach(inp => {
            const name = inp.dataset.param;
            if (name.startsWith('_t')) transform[name.slice(1)] = parseFloat(inp.value) || 0;
            else params[name] = (inp.tagName === 'SELECT') ? inp.value : (parseFloat(inp.value) || 0);
        });
        if (editingBodyId) {
            const id = editingBodyId;
            fetch('/api/cad/bodies/' + id, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ params, transform }) })
                .then(r => r.json().then(data => ({ ok: r.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok) { alert(data.error || 'Update failed'); return; }
                    sceneData = data; renderScene(data); refreshMeshWarning();
                    window.cancelEditBody();
                });
        } else {
            postBody({ type, params, transform, role: 'obstacle' });
        }
    };

    window.applyBoolean = function() {
        const a = document.getElementById('bool-a-select').value;
        const b = document.getElementById('bool-b-select').value;
        const op = document.getElementById('bool-op-select').value;
        if (!a || !b || a === b) { alert('Pick two different bodies'); return; }
        fetch('/api/cad/bodies/boolean', { method: 'POST', body: new URLSearchParams({ a, b, op }) })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => { if (!ok) { alert(data.error || 'Boolean failed'); return; } sceneData = data; renderScene(data); refreshMeshWarning(); });
    };

    window.applyFillet = function() {
        const body = document.getElementById('fillet-body-select').value;
        const mode = document.getElementById('fillet-mode-select').value;
        const radius = parseFloat(document.getElementById('fillet-radius').value) || 0.02;
        if (!body) { alert('Pick a body'); return; }
        postBody({ type: mode, params: { body, radius }, role: 'obstacle' });
    };

    window.uploadCadFile = function(inputEl) {
        const file = inputEl.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        fetch('/api/cad/upload', { method: 'POST', body: formData })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => { if (!ok) { alert(data.error || 'Import failed'); return; } sceneData = data; renderScene(data); refreshMeshWarning(); });
        inputEl.value = '';
    };

    window.exportStl = function() { window.location.href = '/api/cad/export-stl'; };
    window.exportStep = function() { window.location.href = '/api/cad/export-step'; };

    function postBody(body) {
        fetch('/api/cad/bodies', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(({ ok, data }) => { if (!ok) { alert(data.error || 'Operation failed'); return; } sceneData = data; renderScene(data); refreshMeshWarning(); });
    }

    function refreshMeshWarning() {
        if (window.onGeometryChanged) window.onGeometryChanged();
    }

    // ---------------------------------------------------------------
    // 2D Sketch -> Extrude / Revolve (real OCCT surface + extrude/revolve)
    // ---------------------------------------------------------------
    window.startSketch = function() {
        sketchActive = true;
        sketchPoints = [];
        sketchPlane = document.getElementById('sketch-plane-select').value;
        buildConstructionPlane();
        updateSketchPreview();
        document.getElementById('sketch-status').textContent = 'Sketching on ' + sketchPlane.toUpperCase() + ' — click points in viewport';
    };

    window.cancelSketch = function() {
        sketchActive = false;
        sketchPoints = [];
        if (constructionPlaneMesh) { scene.remove(constructionPlaneMesh); constructionPlaneMesh = null; }
        updateSketchPreview();
        document.getElementById('sketch-status').textContent = 'Not sketching';
    };

    window.undoSketchPoint = function() {
        sketchPoints.pop();
        updateSketchPreview();
    };

    function buildConstructionPlane() {
        if (constructionPlaneMesh) scene.remove(constructionPlaneMesh);
        const geo = new THREE.PlaneGeometry(2, 2);
        const mat = new THREE.MeshBasicMaterial({ color: 0x4d8dff, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false });
        constructionPlaneMesh = new THREE.Mesh(geo, mat);
        if (sketchPlane === 'xz') constructionPlaneMesh.rotation.x = -Math.PI / 2;
        else if (sketchPlane === 'yz') constructionPlaneMesh.rotation.y = Math.PI / 2;
        scene.add(constructionPlaneMesh);
    }

    function pointTo3d(p2d) {
        if (sketchPlane === 'xy') return new THREE.Vector3(p2d[0], p2d[1], 0);
        if (sketchPlane === 'xz') return new THREE.Vector3(p2d[0], 0, p2d[1]);
        return new THREE.Vector3(0, p2d[0], p2d[1]);
    }

    function addSketchPoint(worldPoint) {
        let p2d;
        if (sketchPlane === 'xy') p2d = [worldPoint.x, worldPoint.y];
        else if (sketchPlane === 'xz') p2d = [worldPoint.x, worldPoint.z];
        else p2d = [worldPoint.y, worldPoint.z];
        sketchPoints.push(p2d);
        updateSketchPreview();
    }

    function updateSketchPreview() {
        sketchPreviewGroup.clear();
        if (!sketchPoints.length) return;
        const pts3d = sketchPoints.map(pointTo3d);
        pts3d.forEach(p => {
            const dot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffb703 }));
            dot.position.copy(p);
            sketchPreviewGroup.add(dot);
        });
        if (pts3d.length > 1) {
            const geo = new THREE.BufferGeometry().setFromPoints(pts3d);
            sketchPreviewGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffb703 })));
        }
    }

    window.finishSketch = function(mode) {
        if (mode === 'extrude') {
            if (sketchPoints.length < 3) { alert('Need at least 3 points to extrude'); return; }
            const depth = parseFloat(document.getElementById('sketch-depth').value) || 0.15;
            postBody({ type: 'sketch_extrude', params: { points: sketchPoints, plane: sketchPlane, depth }, role: 'obstacle' });
        } else {
            if (sketchPlane === 'xy') { alert('Revolve profiles need a (radius, height) plane — use XZ or YZ, not XY'); return; }
            if (sketchPoints.length < 2) { alert('Need at least 2 points for a revolve profile'); return; }
            if (sketchPoints.some(p => p[0] < 0)) { alert('Revolve profile\'s first coordinate is the radius and must be >= 0'); return; }
            const angle_deg = parseFloat(document.getElementById('sketch-revolve-angle').value) || 360;
            postBody({ type: 'sketch_revolve', params: { points: sketchPoints, angle_deg }, role: 'obstacle' });
        }
        window.cancelSketch();
    };

    // ---------------------------------------------------------------
    // Shading modes & section/clip plane (apply to all bodies)
    // ---------------------------------------------------------------
    window.setShadingMode = function(mode) {
        shadingMode = mode;
        applyShadingModeAll();
        document.querySelectorAll('.shading-mode-btn').forEach(b => {
            if (b.dataset.mode === mode) { b.style.background = 'var(--accent)'; b.style.color = '#0b1220'; }
            else { b.style.background = ''; b.style.color = ''; b.classList.add('panel-overlay'); }
        });
    };

    function applyShadingModeAll() {
        Object.values(bodyMeshes).forEach(bm => {
            bm.wireframeMesh.visible = (shadingMode === 'wireframe' || shadingMode === 'shaded-edges');
            bm.mesh.visible = (shadingMode !== 'wireframe');
            bm.mesh.material.transparent = (shadingMode === 'xray');
            bm.mesh.material.opacity = (shadingMode === 'xray') ? 0.35 : 1.0;
            bm.mesh.material.depthWrite = (shadingMode !== 'xray');
            bm.mesh.material.needsUpdate = true;
        });
    }

    window.toggleSectionPlane = function(enabled) {
        sectionEnabled = enabled;
        renderer.clippingPlanes = enabled ? [sectionPlane] : [];
        Object.values(bodyMeshes).forEach(bm => { bm.mesh.material.clippingPlanes = enabled ? [sectionPlane] : []; bm.mesh.material.needsUpdate = true; });
        document.getElementById('section-slider-row').style.display = enabled ? 'flex' : 'none';
    };

    window.updateSectionOffset = function(value) { sectionPlane.constant = parseFloat(value); };

    window.captureCadScreenshot = function() {
        renderer.render(scene, camera);
        const link = document.createElement('a');
        link.download = 'thapar-cfd-cad-view.png';
        link.href = renderer.domElement.toDataURL('image/png');
        link.click();
    };

    function onWindowResize() {
        const container = document.getElementById('cad-container');
        if (!container) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderGizmo();
    }

    window.reloadCadGeometry = loadCadGeometry;
    window.addEventListener('DOMContentLoaded', init);
})();
