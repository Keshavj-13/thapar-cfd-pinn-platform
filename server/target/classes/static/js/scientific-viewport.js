// Thapar CFD Platform - Three.js Scientific Visualization Viewport
// ParaView-style: simultaneous axial/coronal/sagittal cut-planes streamed over
// one multiplexed binary WebSocket, a switchable colormap bank (Turbo/Viridis/
// Jet/CoolWarm), live point-probe hover tooltips, a real tick-marked legend,
// streamlines, isosurfaces, and PNG export.
(function() {
    let scene, camera, renderer, controls;
    let ws = null;

    const AXES = ['X', 'Y', 'Z'];
    const AXIS_ID = { X: 0, Y: 1, Z: 2 };
    const planes = {}; // axis -> { mesh, texture, w, h }
    let activeColormap = 'turbo';
    let currentMinVal = 0, currentMaxVal = 1;

    // The viewport always renders the domain as a normalized [0,1]^3 cube (same
    // convention the CAD viewport and obstacle placement use), but the solver's
    // actual physical extent (nx*dx, ny*dy, nz*dz) need not be a cube -- the
    // default grid (64x64x16) is a thin slab in z. Point-probes must be scaled
    // from visual cube coordinates to real physical coordinates or they silently
    // land outside the domain and the backend returns a meaningless zero.
    let domainExtent = { x: 1, y: 1, z: 1 };
    function refreshDomainExtent() {
        fetch('/api/sim/config-json')
            .then(res => res.json())
            .then(cfg => {
                domainExtent = { x: cfg.nx * cfg.dx, y: cfg.ny * cfg.dy, z: cfg.nz * cfg.dz };
            })
            .catch(() => {});
    }

    const COLORMAP_GLSL = `
        vec3 turboColormap(float x) {
            const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
            const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503351);
            const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
            const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
            const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
            const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);
            x = clamp(x, 0.0, 1.0);
            vec4 v4 = vec4(1.0, x, x * x, x * x * x);
            vec2 v2 = v4.zw * v4.z;
            return vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2), dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
        }
        vec3 viridisColormap(float x) {
            const vec3 c0 = vec3(0.2777, 0.0054, 0.3341);
            const vec3 c1 = vec3(0.1050, 1.4046, 1.3845);
            const vec3 c2 = vec3(-0.3308, 0.2148, 0.0952);
            const vec3 c3 = vec3(-4.6342, -5.7991, -19.3324);
            const vec3 c4 = vec3(6.2282, 14.1799, 56.6905);
            const vec3 c5 = vec3(4.7763, -13.7451, -65.3529);
            const vec3 c6 = vec3(-5.4354, 4.6458, 26.3124);
            x = clamp(x, 0.0, 1.0);
            return c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6)))));
        }
        vec3 jetColormap(float x) {
            x = clamp(x, 0.0, 1.0);
            float r = clamp(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0);
            float g = clamp(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0);
            float b = clamp(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0);
            return vec3(r, g, b);
        }
        vec3 coolwarmColormap(float x) {
            vec3 cool = vec3(0.230, 0.299, 0.754);
            vec3 mid = vec3(0.865, 0.865, 0.865);
            vec3 warm = vec3(0.706, 0.016, 0.150);
            x = clamp(x, 0.0, 1.0);
            return x < 0.5 ? mix(cool, mid, x * 2.0) : mix(mid, warm, (x - 0.5) * 2.0);
        }
        vec3 applyColormap(float x, int mode) {
            if (mode == 1) return viridisColormap(x);
            if (mode == 2) return jetColormap(x);
            if (mode == 3) return coolwarmColormap(x);
            return turboColormap(x);
        }
    `;

    function init() {
        const container = document.getElementById('scientific-container');
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e17);

        camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
        camera.position.set(1.3, 1.1, 1.9);

        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0.5, 0.5, 0.5);

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const boxGeo = new THREE.BoxGeometry(1, 1, 1);
        const boxEdges = new THREE.EdgesGeometry(boxGeo);
        const boxMat = new THREE.LineBasicMaterial({ color: 0x475569, linewidth: 1 });
        const domainBox = new THREE.LineSegments(boxEdges, boxMat);
        domainBox.position.set(0.5, 0.5, 0.5);
        scene.add(domainBox);

        const axes = new THREE.AxesHelper(0.3);
        scene.add(axes);

        AXES.forEach(axis => createSlicePlane(axis, 64, 64));
        initWebSocket();
        refreshDomainExtent();
        initProbeInteraction(container);
        window.addEventListener('resize', onWindowResize, false);
        animate();
    }

    function colormapIndex(name) {
        return { turbo: 0, viridis: 1, jet: 2, coolwarm: 3 }[name] ?? 0;
    }

    function createSlicePlane(axis, w, h) {
        const existing = planes[axis];
        if (existing) scene.remove(existing.mesh);

        const planeGeo = new THREE.PlaneGeometry(1, 1);
        const initialData = new Float32Array(w * h);
        const texture = new THREE.DataTexture(initialData, w, h, THREE.RedFormat, THREE.FloatType);
        texture.needsUpdate = true;

        const material = new THREE.ShaderMaterial({
            uniforms: {
                uTexture: { value: texture },
                uMin: { value: 0.0 },
                uMax: { value: 1.0 },
                uColormap: { value: colormapIndex(activeColormap) },
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D uTexture;
                uniform float uMin;
                uniform float uMax;
                uniform int uColormap;
                varying vec2 vUv;
                ${COLORMAP_GLSL}
                void main() {
                    float val = texture2D(uTexture, vUv).r;
                    float norm = (val - uMin) / max(uMax - uMin, 1e-6);
                    vec3 col = applyColormap(norm, uColormap);
                    gl_FragColor = vec4(col, 0.95);
                }
            `,
            side: THREE.DoubleSide,
            transparent: true,
        });

        const mesh = new THREE.Mesh(planeGeo, material);
        mesh.userData = { axis };
        positionPlane(mesh, axis, 0.5);
        scene.add(mesh);
        planes[axis] = { mesh, texture, w, h, ratio: 0.5 };
    }

    function positionPlane(mesh, axis, ratio) {
        mesh.rotation.set(0, 0, 0);
        if (axis === 'X') {
            mesh.rotation.y = Math.PI / 2;
            mesh.position.set(ratio, 0.5, 0.5);
        } else if (axis === 'Y') {
            mesh.rotation.x = Math.PI / 2;
            mesh.position.set(0.5, ratio, 0.5);
        } else {
            mesh.position.set(0.5, 0.5, ratio);
        }
    }

    // ---------------------------------------------------------------
    // Binary WebSocket stream (multiplexed: each packet tagged with axis id)
    // ---------------------------------------------------------------
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/cfd-stream`;

        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            const statusBadge = document.getElementById('stream-status-badge');
            if (statusBadge) {
                statusBadge.textContent = "STREAM LIVE (A100)";
                statusBadge.className = "px-2 py-0.5 rounded text-xs font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            }
        };
        ws.onmessage = (event) => {
            if (!(event.data instanceof ArrayBuffer)) return;
            handleBinarySlice(event.data);
        };
        ws.onclose = () => {
            const statusBadge = document.getElementById('stream-status-badge');
            if (statusBadge) {
                statusBadge.textContent = "DISCONNECTED";
                statusBadge.className = "px-2 py-0.5 rounded text-xs font-mono font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30";
            }
            setTimeout(initWebSocket, 2000);
        };
    }

    // Results viewing mode: 'live' follows the WebSocket stream as normal;
    // 'checkpoint' shows a static saved snapshot fetched on demand and
    // ignores incoming live frames until returned to live (see setResultsViewMode).
    let resultsViewMode = 'live';
    let checkpointStep = null;

    function handleBinarySlice(buffer) {
        if (resultsViewMode !== 'live') return;
        const view = new DataView(buffer);
        const magic = view.getInt32(0, true);
        if (magic !== 0x54485052) return; // "THPR"

        const width = view.getInt32(4, true);
        const height = view.getInt32(8, true);
        const axisId = view.getInt32(12, true);
        const minVal = view.getFloat32(16, true);
        const maxVal = view.getFloat32(20, true);
        const floatArray = new Float32Array(buffer, 24, width * height);

        const axis = AXES[axisId] || 'Z';
        let plane = planes[axis];
        if (!plane || plane.w !== width || plane.h !== height) {
            createSlicePlane(axis, width, height);
            plane = planes[axis];
        }

        plane.texture.image.data.set(floatArray);
        plane.texture.needsUpdate = true;
        plane.lastData = floatArray;
        plane.mesh.material.uniforms.uMin.value = minVal;
        plane.mesh.material.uniforms.uMax.value = maxVal;

        currentMinVal = minVal;
        currentMaxVal = maxVal;
        updateLegend(minVal, maxVal);
        updateFieldStats(floatArray, minVal, maxVal);
    }

    function updateFieldStats(arr, minVal, maxVal) {
        if (!arr || !arr.length) return;
        let sum = 0;
        for (let i = 0; i < arr.length; i++) sum += arr[i];
        const mean = sum / arr.length;
        let sqSum = 0;
        for (let i = 0; i < arr.length; i++) { const d = arr[i] - mean; sqSum += d * d; }
        const stddev = Math.sqrt(sqSum / arr.length);
        const summary = `min ${minVal.toFixed(4)} · max ${maxVal.toFixed(4)} · mean ${mean.toFixed(4)} (n=${arr.length})`;

        const inlineEl = document.getElementById('field-stats-label');
        if (inlineEl) inlineEl.textContent = summary;

        const panelEl = document.getElementById('stats-panel-content');
        if (panelEl) {
            const fieldSelect = document.getElementById('field-select');
            const fieldName = fieldSelect ? fieldSelect.options[fieldSelect.selectedIndex].text : 'Field';
            panelEl.innerHTML = `
                <div class="text-[10px] font-mono" style="color: var(--text-muted);">${fieldName}</div>
                <div class="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px] font-mono mt-1">
                    <span style="color: var(--text-muted);">Min</span><span style="color: var(--text-primary);">${minVal.toFixed(5)}</span>
                    <span style="color: var(--text-muted);">Max</span><span style="color: var(--text-primary);">${maxVal.toFixed(5)}</span>
                    <span style="color: var(--text-muted);">Mean</span><span style="color: var(--text-primary);">${mean.toFixed(5)}</span>
                    <span style="color: var(--text-muted);">Std Dev</span><span style="color: var(--text-primary);">${stddev.toFixed(5)}</span>
                    <span style="color: var(--text-muted);">Range</span><span style="color: var(--text-primary);">${(maxVal - minVal).toFixed(5)}</span>
                    <span style="color: var(--text-muted);">N cells</span><span style="color: var(--text-primary);">${arr.length}</span>
                </div>`;
        }
    }

    const FIELD_TO_BRIDGE_KEY = { VELOCITY: 'speed', PRESSURE: 'pressure', TEMPERATURE: 'temperature', TURBULENCE: 'nu_t', VOF_ALPHA: 'alpha' };

    /** Switches the Results viewport between following the live GPU stream
     *  and displaying a static saved checkpoint (fetched via the Java-side
     *  raw-.bin slice reader, not the live WebSocket). */
    window.setResultsViewMode = function(mode, step) {
        resultsViewMode = mode;
        checkpointStep = mode === 'checkpoint' ? step : null;
        if (mode === 'live') return; // next WS frame naturally repaints everything

        const fieldSelect = document.getElementById('field-select');
        const active = fieldSelect ? fieldSelect.value : 'VELOCITY';
        const bridgeField = FIELD_TO_BRIDGE_KEY[active] || 'speed';

        AXES.forEach(axis => {
            const plane = planes[axis];
            if (!plane || !plane.mesh.visible) return;
            const slider = document.getElementById('slice-slider-' + axis);
            const idx = slider ? parseInt(slider.value) : 0;
            fetch(`/api/sim/checkpoints/${step}/slice?field=${bridgeField}&axis=${axis}&index=${idx}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) { console.warn('Checkpoint slice error', data.error); return; }
                    const arr = new Float32Array(data.data);
                    if (plane.w !== data.width || plane.h !== data.height) {
                        createSlicePlane(axis, data.width, data.height);
                    }
                    const p = planes[axis];
                    p.texture.image.data.set(arr);
                    p.texture.needsUpdate = true;
                    p.mesh.material.uniforms.uMin.value = data.min;
                    p.mesh.material.uniforms.uMax.value = data.max;
                    updateLegend(data.min, data.max);
                    updateFieldStats(arr, data.min, data.max);
                });
        });
    };

    // ---------------------------------------------------------------
    // Legend with real tick marks
    // ---------------------------------------------------------------
    function updateLegend(minVal, maxVal) {
        const ticksEl = document.getElementById('legend-ticks');
        if (!ticksEl) return;
        const nTicks = 5;
        let html = '';
        for (let i = 0; i < nTicks; i++) {
            const v = minVal + (maxVal - minVal) * (i / (nTicks - 1));
            html += `<span>${v.toFixed(v >= 100 ? 0 : 3)}</span>`;
        }
        ticksEl.innerHTML = html;
    }

    // ---------------------------------------------------------------
    // Colormap switcher & multi cut-plane visibility toggles
    // ---------------------------------------------------------------
    window.setColormap = function(name) {
        activeColormap = name;
        Object.values(planes).forEach(p => { p.mesh.material.uniforms.uColormap.value = colormapIndex(name); });
        document.getElementById('legend-gradient').className = 'legend-gradient cmap-' + name;
        document.querySelectorAll('.colormap-btn').forEach(b => {
            b.classList.toggle('ring-2', b.dataset.cmap === name);
            b.classList.toggle('ring-sky-400', b.dataset.cmap === name);
        });
    };

    window.toggleSlicePlane = function(axis, visible) {
        if (planes[axis]) planes[axis].mesh.visible = visible;
        fetch(`/api/sim/slice?axis=${axis}&index=${getSliderIndex(axis)}&visible=${visible}`, { method: 'POST' });
    };

    window.onSlicePositionChange = function(axis) {
        const slider = document.getElementById('slice-slider-' + axis);
        const idx = parseInt(slider.value);
        document.getElementById('slice-index-label-' + axis).textContent = idx;

        let maxIdx = parseInt(slider.max);
        const ratio = (idx + 0.5) / (maxIdx + 1);
        if (planes[axis]) positionPlane(planes[axis].mesh, axis, ratio);

        fetch(`/api/sim/slice?axis=${axis}&index=${idx}&visible=true`, { method: 'POST' });
    };

    function getSliderIndex(axis) {
        const slider = document.getElementById('slice-slider-' + axis);
        return slider ? parseInt(slider.value) : 0;
    }

    // ---------------------------------------------------------------
    // Live point-probe hover tooltip (raycast against visible slice planes)
    // ---------------------------------------------------------------
    let lastHoverPhysicalPoint = null;

    function initProbeInteraction(container) {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('probe-tooltip');
        let probeInFlight = false;
        let lastProbeTime = 0;

        container.tabIndex = 0; // make the canvas container focusable so it can receive keydown
        container.addEventListener('keydown', (event) => {
            if (event.key.toLowerCase() === 'p' && lastHoverPhysicalPoint && window.setPinnedProbe) {
                window.setPinnedProbe(lastHoverPhysicalPoint.x, lastHoverPhysicalPoint.y, lastHoverPhysicalPoint.z);
            }
        });
        container.addEventListener('mouseenter', () => container.focus());

        container.addEventListener('mousemove', (event) => {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const targets = Object.values(planes).filter(p => p.mesh.visible).map(p => p.mesh);
            const hits = raycaster.intersectObjects(targets);

            if (hits.length === 0) {
                if (tooltip) tooltip.classList.add('hidden');
                return;
            }

            const hit = hits[0];
            const p = hit.point; // visual [0,1]^3 cube coordinates
            // Scale to the solver's real physical extent (see domainExtent above) --
            // the probe endpoint takes physical meters, not normalized cube coords.
            const px = p.x * domainExtent.x, py = p.y * domainExtent.y, pz = p.z * domainExtent.z;
            lastHoverPhysicalPoint = { x: px, y: py, z: pz };

            if (tooltip) {
                tooltip.classList.remove('hidden');
                tooltip.style.left = (event.clientX - rect.left + 14) + 'px';
                tooltip.style.top = (event.clientY - rect.top + 14) + 'px';
                tooltip.innerHTML = `<b>(${px.toFixed(3)}, ${py.toFixed(3)}, ${pz.toFixed(3)}) m</b><br><span class="text-slate-400">probing... (press P to pin)</span>`;
            }

            const now = performance.now();
            if (!probeInFlight && now - lastProbeTime > 80) {
                probeInFlight = true;
                lastProbeTime = now;
                fetch(`/api/sim/probe?x=${px}&y=${py}&z=${pz}`)
                    .then(res => res.json())
                    .then(data => {
                        probeInFlight = false;
                        if (!tooltip || tooltip.classList.contains('hidden')) return;
                        const fieldSelect = document.getElementById('field-select');
                        const active = fieldSelect ? fieldSelect.value : 'VELOCITY';
                        const probeKey = { VELOCITY: 'speed', PRESSURE: 'pressure', TEMPERATURE: 'temperature', TURBULENCE: 'nu_t', VOF_ALPHA: 'alpha' }[active];
                        const val = (probeKey && data[probeKey] !== undefined) ? Number(data[probeKey]).toFixed(5) : 'n/a';
                        tooltip.innerHTML = `<b>(${px.toFixed(3)}, ${py.toFixed(3)}, ${pz.toFixed(3)}) m</b><br>` +
                            `<span class="text-sky-400 font-bold">${val}</span> <span class="text-slate-400">${active}</span>`;
                    })
                    .catch(() => { probeInFlight = false; });
            }
        });

        container.addEventListener('mouseleave', () => {
            if (tooltip) tooltip.classList.add('hidden');
        });
    }

    // ---------------------------------------------------------------
    // Streamlines & isosurface
    // ---------------------------------------------------------------
    let streamlineGroup = new THREE.Group();
    let isosurfaceGroup = new THREE.Group();
    let showStreamlines = false;
    let showIsosurface = false;

    window.toggleStreamlines = function() {
        showStreamlines = !showStreamlines;
        if (!showStreamlines) scene.remove(streamlineGroup);
        else fetchStreamlines();
    };

    window.toggleIsosurface = function(isoval = 0.5) {
        showIsosurface = !showIsosurface;
        if (!showIsosurface) scene.remove(isosurfaceGroup);
        else fetchIsosurface(isoval);
    };

    function fetchStreamlines() {
        fetch('/api/sim/streamlines')
            .then(res => res.json())
            .then(data => {
                scene.remove(streamlineGroup);
                streamlineGroup = new THREE.Group();
                if (!data.lines) return;
                data.lines.forEach(line => {
                    const points = line.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
                    const curve = new THREE.CatmullRomCurve3(points);
                    const tubeGeo = new THREE.TubeGeometry(curve, points.length * 2, 0.004, 6, false);
                    const tubeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.2, roughness: 0.3 });
                    streamlineGroup.add(new THREE.Mesh(tubeGeo, tubeMat));
                });
                scene.add(streamlineGroup);
            })
            .catch(err => console.warn("Streamline fetch error", err));
    }

    function fetchIsosurface(isoval) {
        fetch('/api/sim/isosurface?isovalue=' + isoval)
            .then(res => res.json())
            .then(data => {
                scene.remove(isosurfaceGroup);
                isosurfaceGroup = new THREE.Group();
                if (!data.vertices || data.vertices.length === 0) return;
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
                if (data.normals && data.normals.length > 0) geo.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
                else geo.computeVertexNormals();
                geo.setIndex(data.faces);
                const mat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.2, metalness: 0.4, transparent: true, opacity: 0.65, side: THREE.DoubleSide });
                isosurfaceGroup.add(new THREE.Mesh(geo, mat));
                scene.add(isosurfaceGroup);
            })
            .catch(err => console.warn("Isosurface fetch error", err));
    }

    window.captureScientificScreenshot = function() {
        renderer.render(scene, camera);
        const link = document.createElement('a');
        link.download = 'thapar-cfd-field-view.png';
        link.href = renderer.domElement.toDataURL('image/png');
        link.click();
    };

    function onWindowResize() {
        const container = document.getElementById('scientific-container');
        if (!container) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    window.addEventListener('DOMContentLoaded', init);
})();
