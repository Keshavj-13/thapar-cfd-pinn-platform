package com.thapar.cfd.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.thapar.cfd.model.SimulationConfig;
import com.thapar.cfd.service.CadPreprocessService;
import com.thapar.cfd.service.CfdEngineService;
import com.thapar.cfd.service.CheckpointService;
import com.thapar.cfd.service.CustomEquationService;
import com.thapar.cfd.service.ParaViewExportService;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

@Controller
public class CfdDashboardController {

    private final CfdEngineService engineService;
    private final CadPreprocessService cadService;
    private final ParaViewExportService paraViewService;
    private final CheckpointService checkpointService;
    private final CustomEquationService customEquationService;
    private final ObjectMapper mapper = new ObjectMapper();

    /** Whether the current GPU-resident obstacle mask matches the current CAD scene
     *  + resolution -- surfaced to the UI so "Mesh" status is never a silent lie. */
    private volatile boolean meshUpToDate = false;

    public CfdDashboardController(CfdEngineService engineService,
                                  CadPreprocessService cadService,
                                  ParaViewExportService paraViewService,
                                  CheckpointService checkpointService,
                                  CustomEquationService customEquationService) {
        this.engineService = engineService;
        this.cadService = cadService;
        this.paraViewService = paraViewService;
        this.checkpointService = checkpointService;
        this.customEquationService = customEquationService;
        this.engineService.setCheckpointCallback(checkpointService::capture);
        initDefaultCad();
    }

    private void initDefaultCad() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            cadService.addBody("cylinder", mapper.readTree("{\"radius\":0.1,\"height\":0.8,\"axis\":\"z\"}"),
                    mapper.readTree("{\"tx\":0.3,\"ty\":0.5,\"tz\":0.4}"), "obstacle", "Cylinder");
            cadService.regeneratePreview(cfg.getNx(), cfg.getNy(), cfg.getNz(),
                    cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
            // NOTE: intentionally NOT pushed to the GPU here -- the obstacle mask is
            // only committed via /api/mesh/generate (explicit Mesh-stage action), so
            // even the very first page load never depends on an implicit GPU write.
        } catch (Exception e) {
            // Non-fatal: an empty/default scene is a valid starting point.
        }
    }

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("config", engineService.getCurrentConfig());
        model.addAttribute("diag", engineService.getDiagnostics());
        return "index";
    }

    @GetMapping("/api/sim/diagnostics")
    public String getDiagnosticsFragment(Model model) {
        model.addAttribute("diag", engineService.getDiagnostics());
        model.addAttribute("config", engineService.getCurrentConfig());
        return "fragments/diagnostics :: diagFragment";
    }

    /** Same diagnostics as raw JSON for JS-side consumers (convergence chart,
     *  case-status compute-state chip) that shouldn't have to scrape rendered HTML. */
    @GetMapping(value = "/api/sim/diagnostics.json", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public com.thapar.cfd.model.SolverDiagnostics getDiagnosticsJson() {
        return engineService.getDiagnostics();
    }

    @PostMapping("/api/sim/start")
    @ResponseBody
    public String start() {
        engineService.startContinuousSimulation();
        return "RUNNING";
    }

    @PostMapping("/api/sim/pause")
    @ResponseBody
    public String pause() {
        engineService.pauseSimulation();
        return "PAUSED";
    }

    @PostMapping("/api/sim/step")
    @ResponseBody
    public String step(@RequestParam(defaultValue = "10") int steps) {
        engineService.singleStep(steps);
        return "STEPPED";
    }

    @PostMapping("/api/sim/reset")
    @ResponseBody
    public String reset() {
        engineService.resetSimulation();
        checkpointService.clear();
        return "RESET";
    }

    /** The "actual run" mode: executes exactly `steps` more steps, capturing a
     *  checkpoint every `checkpointInterval` steps (0 disables checkpointing),
     *  then auto-stops -- distinct from "Start Live Preview" (unbounded, no
     *  checkpoints, manual stop). Still requires this explicit call; opening
     *  the page never triggers it. */
    @PostMapping("/api/sim/run")
    @ResponseBody
    public ResponseEntity<String> runToCompletion(
            @RequestParam long steps,
            @RequestParam(defaultValue = "0") long checkpointInterval) {
        try {
            engineService.startRun(steps, checkpointInterval);
            return ResponseEntity.ok("{\"status\":\"running\",\"targetStep\":" + engineService.getDiagnostics().getTargetStep() + "}");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @GetMapping(value = "/api/sim/checkpoints", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public String listCheckpoints() {
        var arr = mapper.createArrayNode();
        for (var c : checkpointService.list()) {
            var node = mapper.createObjectNode();
            node.put("step", c.step());
            node.put("physicalTime", c.physicalTime());
            node.put("timestampMs", c.timestampMs());
            arr.add(node);
        }
        return arr.toString();
    }

    /** Reads a 2D slice out of a saved checkpoint (not the live GPU state) --
     *  what the Results stage's timestep scrubber calls when browsing history
     *  instead of watching the live stream. */
    @GetMapping(value = "/api/sim/checkpoints/{step}/slice", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> getCheckpointSlice(
            @PathVariable long step,
            @RequestParam String field,
            @RequestParam String axis,
            @RequestParam int index) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            float[] data = checkpointService.readSlice(step, field.toLowerCase(), axis, index, cfg.getNx(), cfg.getNy(), cfg.getNz());
            float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
            for (float v : data) { if (v < min) min = v; if (v > max) max = v; }
            int w = "x".equalsIgnoreCase(axis) ? cfg.getNy() : cfg.getNx();
            int h = "z".equalsIgnoreCase(axis) ? cfg.getNy() : cfg.getNz();
            var node = mapper.createObjectNode();
            node.put("width", w); node.put("height", h);
            node.put("min", min); node.put("max", max);
            var arr = mapper.createArrayNode();
            for (float v : data) arr.add(v);
            node.set("data", arr);
            return ResponseEntity.ok(node.toString());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @PostMapping("/api/sim/field")
    @ResponseBody
    public String setField(@RequestParam String field) {
        engineService.getCurrentConfig().setActiveField(field);
        engineService.publishCurrentSlice();
        return field;
    }

    @PostMapping("/api/sim/slice")
    @ResponseBody
    public String setSlice(@RequestParam String axis, @RequestParam int index,
                            @RequestParam(defaultValue = "true") boolean visible) {
        SimulationConfig cfg = engineService.getCurrentConfig();
        switch (axis.toUpperCase()) {
            case "X" -> { cfg.setSliceIndexX(index); cfg.setShowSliceX(visible); }
            case "Y" -> { cfg.setSliceIndexY(index); cfg.setShowSliceY(visible); }
            default -> { cfg.setSliceIndexZ(index); cfg.setShowSliceZ(visible); }
        }
        engineService.publishCurrentSlice();
        return "OK";
    }

    // -----------------------------------------------------------------
    // Parametric CAD: multi-body scene (real geometry creation/editing,
    // not just "upload a mesh"). See CadPreprocessService + tools/cad_kernel_freecad.py.
    // Every mutation regenerates the client-facing preview immediately but
    // NEVER touches the GPU solver -- that only happens in /api/mesh/generate.
    // -----------------------------------------------------------------

    @GetMapping(value = "/api/cad/scene", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public String getScene() {
        return cadService.getSceneJson();
    }

    @GetMapping(value = "/api/sim/cad-geometry", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public String getCadGeometry() {
        return cadService.getLastGeometryJson();
    }

    private ResponseEntity<String> regenerateAndRespond() {
        SimulationConfig cfg = engineService.getCurrentConfig();
        CadPreprocessService.RegenResult res = cadService.regeneratePreview(
                cfg.getNx(), cfg.getNy(), cfg.getNz(),
                cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
        meshUpToDate = false; // geometry changed -- any previously generated mesh is now stale
        if (!res.ok()) {
            return ResponseEntity.unprocessableEntity()
                    .body("{\"error\":" + mapper.valueToTree(res.error()) + "}");
        }
        return ResponseEntity.ok(res.geometryJson());
    }

    @PostMapping(value = "/api/cad/bodies", consumes = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> addBody(@RequestBody String rawJson) {
        try {
            JsonNode body = mapper.readTree(rawJson);
            String type = body.get("type").asText();
            JsonNode params = body.has("params") ? body.get("params") : mapper.createObjectNode();
            JsonNode transform = body.has("transform") ? body.get("transform") : null;
            String role = body.has("role") ? body.get("role").asText() : "obstacle";
            String label = body.has("label") ? body.get("label").asText() : null;
            String id = cadService.addBody(type, params, transform, role, label);
            ResponseEntity<String> resp = regenerateAndRespond();
            if (!resp.getStatusCode().is2xxSuccessful()) {
                cadService.deleteBody(id); // roll back so a failed body never lingers in the scene
                return resp;
            }
            return resp;
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @PutMapping(value = "/api/cad/bodies/{id}", consumes = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> updateBody(@PathVariable String id, @RequestBody String rawJson) {
        try {
            JsonNode body = mapper.readTree(rawJson);
            JsonNode params = body.has("params") ? body.get("params") : null;
            JsonNode transform = body.has("transform") ? body.get("transform") : null;
            if (!cadService.updateBody(id, params, transform)) {
                return ResponseEntity.notFound().build();
            }
            return regenerateAndRespond();
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @DeleteMapping("/api/cad/bodies/{id}")
    @ResponseBody
    public ResponseEntity<String> deleteBody(@PathVariable String id) {
        if (!cadService.deleteBody(id)) {
            return ResponseEntity.status(409).body("{\"error\":\"Body is referenced by a boolean operation, or does not exist -- delete the dependent boolean first.\"}");
        }
        return regenerateAndRespond();
    }

    @PostMapping(value = "/api/cad/bodies/boolean")
    @ResponseBody
    public ResponseEntity<String> booleanOp(@RequestParam String a, @RequestParam String b,
                                             @RequestParam String op, @RequestParam(required = false) String label) {
        if (!cadService.hasBody(a) || !cadService.hasBody(b)) {
            return ResponseEntity.badRequest().body("{\"error\":\"Both bodies must exist\"}");
        }
        String id = cadService.addBoolean(a, b, op, label);
        ResponseEntity<String> resp = regenerateAndRespond();
        if (!resp.getStatusCode().is2xxSuccessful()) {
            cadService.deleteBody(id);
        }
        return resp;
    }

    @PostMapping("/api/cad/upload")
    @ResponseBody
    public ResponseEntity<String> uploadCad(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) return ResponseEntity.badRequest().body("{\"error\":\"Empty file\"}");
        try {
            String filename = Path.of(file.getOriginalFilename()).getFileName().toString();
            File dest = new File(cadService.getUploadDir(), System.currentTimeMillis() + "_" + filename);
            Files.copy(file.getInputStream(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
            cadService.importFile(dest, filename);
            return regenerateAndRespond();
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @GetMapping("/api/cad/export-stl")
    public ResponseEntity<Resource> exportCadStl() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            File stl = cadService.exportStl(cfg.getNx(), cfg.getNy(), cfg.getNz(),
                    cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"flowstudio_geometry.stl\"")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .body(new FileSystemResource(stl));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    /** Real B-Rep STEP export (true CAD interchange format) -- possible because
     *  the kernel underneath is genuine OpenCASCADE, not a mesh-only library. */
    @GetMapping("/api/cad/export-step")
    public ResponseEntity<Resource> exportCadStep() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            File step = cadService.exportStep(cfg.getNx(), cfg.getNy(), cfg.getNz(),
                    cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"flowstudio_geometry.step\"")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .body(new FileSystemResource(step));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @PostMapping("/api/cad/clear")
    @ResponseBody
    public ResponseEntity<String> clearScene() {
        cadService.clearScene();
        return regenerateAndRespond();
    }

    /** Explicit Mesh-stage action: commits the current CAD scene's voxelized Brinkman
     *  mask to the GPU solver (allocating/reinitializing it if needed). This is the
     *  ONLY place geometry edits ever reach the GPU -- never on page load, never on a
     *  bare CAD edit. Still does not start compute (see /api/sim/start for that). */
    @PostMapping(value = "/api/mesh/generate", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> generateMesh() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            CadPreprocessService.RegenResult res = cadService.regeneratePreview(
                    cfg.getNx(), cfg.getNy(), cfg.getNz(),
                    cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
            if (!res.ok()) {
                return ResponseEntity.unprocessableEntity().body("{\"error\":" + mapper.valueToTree(res.error()) + "}");
            }
            byte[] maskBytes = cadService.getLatestMaskBytes();
            engineService.setObstacleMask(maskBytes);
            meshUpToDate = true;

            long cells = (long) cfg.getNx() * cfg.getNy() * cfg.getNz();
            float lx = cfg.getNx() * cfg.getDx(), ly = cfg.getNy() * cfg.getDy(), lz = cfg.getNz() * cfg.getDz();
            float maxExtent = Math.max(lx, Math.max(ly, lz));
            float minExtent = Math.min(lx, Math.min(ly, lz));
            boolean anisotropic = (maxExtent / Math.max(minExtent, 1e-9f)) > 1.5f;

            var out = mapper.createObjectNode();
            out.put("status", "meshed");
            out.put("cells", cells);
            out.put("nx", cfg.getNx()); out.put("ny", cfg.getNy()); out.put("nz", cfg.getNz());
            out.put("domain_lx", lx); out.put("domain_ly", ly); out.put("domain_lz", lz);
            out.put("anisotropic_warning", anisotropic);
            return ResponseEntity.ok(out.toString());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Custom Equations (Expert): evaluates a user-typed spatial expression sigma(x,y,z)
     *  over the mesh grid without touching the GPU solver, so users can check the field's
     *  min/max/mean and how many cells it affects before committing it. */
    @PostMapping(value = "/api/mesh/custom-equation/preview", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> previewCustomEquation(@RequestBody java.util.Map<String, Object> body) {
        try {
            String expr = String.valueOf(body.get("expression"));
            SimulationConfig cfg = engineService.getCurrentConfig();
            CustomEquationService.Result r = customEquationService.evaluateField(expr, cfg);
            var out = mapper.createObjectNode();
            out.put("status", "ok");
            out.put("min", r.min());
            out.put("max", r.max());
            out.put("mean", r.mean());
            out.put("nonzeroCells", r.nonzeroCells());
            out.put("totalCells", r.totalCells());
            return ResponseEntity.ok(out.toString());
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("{\"error\":\"" + safeMsg(e) + "\"}");
        }
    }

    /** Custom Equations (Expert): evaluates sigma(x,y,z), merges it with the current
     *  geometry-derived Brinkman mask per `mode` (add/max/replace), and uploads the
     *  combined field to the live GPU solver -- the same channel real CAD-derived
     *  obstacles use, so this takes effect on the very next step. */
    @PostMapping(value = "/api/mesh/custom-equation/apply", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> applyCustomEquation(@RequestBody java.util.Map<String, Object> body) {
        try {
            String expr = String.valueOf(body.get("expression"));
            String mode = body.containsKey("mode") ? String.valueOf(body.get("mode")) : "add";
            SimulationConfig cfg = engineService.getCurrentConfig();
            byte[] baseMask = null;
            try { baseMask = cadService.getLatestMaskBytes(); } catch (Exception ignored) { /* no geometry mask yet */ }
            byte[] combined = customEquationService.applyToMask(expr, mode, cfg, baseMask);
            engineService.setObstacleMask(combined);
            var out = mapper.createObjectNode();
            out.put("status", "applied");
            out.put("expression", expr);
            out.put("mode", mode);
            return ResponseEntity.ok(out.toString());
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("{\"error\":\"" + safeMsg(e) + "\"}");
        }
    }

    private String safeMsg(Exception e) {
        String m = e.getMessage() == null ? e.toString() : e.getMessage();
        return m.replace("\"", "'").replace("\n", " ");
    }

    @GetMapping("/api/mesh/status")
    @ResponseBody
    public ResponseEntity<String> meshStatus() {
        return ResponseEntity.ok("{\"upToDate\":" + meshUpToDate + "}");
    }

    /** Mesh stage: updates Cartesian grid resolution / physical domain size independently
     *  of the Physics/Materials settings (unlike the Basic-mode quick-setup wizard, which
     *  couples resolution to Re for beginners). Reinitializes the solver (idle -- no
     *  compute) and marks the mesh stale until /api/mesh/generate is explicitly run. */
    @PostMapping("/api/mesh/settings")
    @ResponseBody
    public ResponseEntity<String> setMeshSettings(
            @RequestParam int nx, @RequestParam int ny, @RequestParam int nz,
            @RequestParam float lx, @RequestParam float ly, @RequestParam float lz,
            @RequestParam(required = false) Integer nlevel,
            @RequestParam(required = false) Integer mgIterations) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            cfg.setNx(nx); cfg.setNy(ny); cfg.setNz(nz);
            cfg.setDx(lx / nx); cfg.setDy(ly / ny); cfg.setDz(lz / nz);
            if (nlevel != null) cfg.setNlevel(nlevel);
            if (mgIterations != null) cfg.setMgIterations(mgIterations);
            engineService.initializeSolver(cfg);
            engineService.publishCurrentSlice();
            meshUpToDate = false;
            return ResponseEntity.ok(CfdEngineService.buildEngineJson(cfg));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Physics stage: characteristic velocity scale (Re is DERIVED from this and the
     *  Materials stage's viscosity, never set independently -- avoids Physics and
     *  Materials silently disagreeing about Re) plus multiphysics module toggles. */
    @PostMapping("/api/sim/physics")
    @ResponseBody
    public ResponseEntity<String> setPhysics(
            @RequestParam(required = false) Float ub,
            @RequestParam(required = false) Boolean enableHeat,
            @RequestParam(required = false) Boolean enableTurbulence,
            @RequestParam(required = false) Boolean enableVof) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            if (ub != null) { cfg.setUb(ub); cfg.getBcZmax().setU(ub); }
            if (enableHeat != null) cfg.setEnableHeat(enableHeat);
            if (enableTurbulence != null) cfg.setEnableTurbulence(enableTurbulence);
            if (enableVof != null) cfg.setEnableVof(enableVof);
            engineService.initializeSolver(cfg);
            engineService.publishCurrentSlice();
            return ResponseEntity.ok(CfdEngineService.buildEngineJson(cfg));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Expert level: direct access to the multiphysics coefficients that Basic/Advanced
     *  never expose (thermal diffusivity, Boussinesq beta/T_ref, gravity vector, VoF
     *  two-phase density/viscosity pairs) -- "expert must retain access to advanced
     *  numerical controls" is not satisfied by toggles alone. */
    @PostMapping("/api/sim/physics-advanced")
    @ResponseBody
    public ResponseEntity<String> setPhysicsAdvanced(
            @RequestParam(required = false) Float thermalDiffusivity,
            @RequestParam(required = false) Float betaThermal,
            @RequestParam(required = false) Float tRef,
            @RequestParam(required = false) Float gx,
            @RequestParam(required = false) Float gy,
            @RequestParam(required = false) Float gz,
            @RequestParam(required = false) Float rho1,
            @RequestParam(required = false) Float rho2,
            @RequestParam(required = false) Float nu1,
            @RequestParam(required = false) Float nu2) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            if (thermalDiffusivity != null) cfg.setThermalDiffusivity(thermalDiffusivity);
            if (betaThermal != null) cfg.setBetaThermal(betaThermal);
            if (tRef != null) cfg.settRef(tRef);
            if (gx != null) cfg.setGx(gx);
            if (gy != null) cfg.setGy(gy);
            if (gz != null) cfg.setGz(gz);
            if (rho1 != null) cfg.setRho1(rho1);
            if (rho2 != null) cfg.setRho2(rho2);
            if (nu1 != null) cfg.setNu1(nu1);
            if (nu2 != null) cfg.setNu2(nu2);
            engineService.initializeSolver(cfg);
            engineService.publishCurrentSlice();
            return ResponseEntity.ok(CfdEngineService.buildEngineJson(cfg));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Materials stage: sets fluid density/viscosity independently of the Physics stage's
     *  Reynolds-number slider (Basic mode), keeping the two mutually consistent. */
    @PostMapping("/api/sim/material")
    @ResponseBody
    public ResponseEntity<String> setMaterial(@RequestParam float rho, @RequestParam float nu) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            cfg.setRho(rho); cfg.setNu(nu);
            engineService.initializeSolver(cfg);
            engineService.publishCurrentSlice();
            return ResponseEntity.ok(CfdEngineService.buildEngineJson(cfg));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    @GetMapping("/api/sim/export-vti")
    public ResponseEntity<Resource> exportVti() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            File vtiFile = paraViewService.exportVti(cfg.getNx(), cfg.getNy(), cfg.getNz(), cfg.getDx(), cfg.getDy(), cfg.getDz());
            Resource resource = new FileSystemResource(vtiFile);

            return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"flowstudio_solution.vti\"")
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(resource);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @GetMapping(value = "/api/sim/streamlines", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> getStreamlines() {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            String json = paraViewService.generateStreamlines(cfg.getNx(), cfg.getNy(), cfg.getNz(), cfg.getDx(), cfg.getDy(), cfg.getDz());
            return ResponseEntity.ok(json);
        } catch (Exception e) {
            return ResponseEntity.ok("{\"num_lines\":0,\"lines\":[]}");
        }
    }

    @GetMapping(value = "/api/sim/isosurface", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> getIsosurface(@RequestParam(defaultValue = "0.5") float isovalue) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            String json = paraViewService.generateIsosurface(cfg.getNx(), cfg.getNy(), cfg.getNz(), cfg.getDx(), cfg.getDy(), cfg.getDz(), isovalue);
            return ResponseEntity.ok(json);
        } catch (Exception e) {
            return ResponseEntity.ok("{\"num_vertices\":0,\"num_faces\":0,\"vertices\":[],\"faces\":[]}");
        }
    }

    @GetMapping(value = "/api/sim/probe", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> probe(@RequestParam float x, @RequestParam float y, @RequestParam float z) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            String json = paraViewService.probePoint(x, y, z, cfg.getNx(), cfg.getNy(), cfg.getNz(), cfg.getDx(), cfg.getDy(), cfg.getDz());
            return ResponseEntity.ok(json);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Professional CAD face-picker: assigns a physical boundary condition (Inlet, Outlet,
     *  No-Slip Wall, Free-Slip Symmetry, Moving Lid) to one of the 6 outer domain patches
     *  and rebuilds the native solver with it applied. */
    @PostMapping("/api/sim/boundary-condition")
    @ResponseBody
    public ResponseEntity<String> setBoundaryCondition(
            @RequestParam String face,
            @RequestParam String type,
            @RequestParam(defaultValue = "0") float u,
            @RequestParam(defaultValue = "0") float v,
            @RequestParam(defaultValue = "0") float w) {
        try {
            engineService.applyBoundaryCondition(face, type, u, v, w);
            return ResponseEntity.ok("{\"status\":\"applied\",\"face\":\"" + face + "\",\"type\":\"" + type + "\"}");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Faculty Expert Mode: returns the exact flat sim_spec.json IR the native UCOF engine consumes. */
    @GetMapping(value = "/api/sim/config-json", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public String getConfigJson() {
        return CfdEngineService.buildEngineJson(engineService.getCurrentConfig());
    }

    /** Faculty Expert Mode: applies a hand-edited sim_spec.json IR directly to the solver. */
    @PostMapping(value = "/api/sim/config-json", consumes = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<String> setConfigJson(@RequestBody String rawJson) {
        try {
            engineService.applyRawJsonConfig(rawJson);
            meshUpToDate = false; // new solver instance -> obstacle mask reset until Mesh stage regenerates it
            return ResponseEntity.ok(CfdEngineService.buildEngineJson(engineService.getCurrentConfig()));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    /** Student Wizard quick-setup: derives a full case (grid, dt, nu) from a Reynolds
     *  number and resolution preset with a built-in CFL safety check. */
    @PostMapping("/api/sim/quick-setup")
    @ResponseBody
    public ResponseEntity<String> quickSetup(
            @RequestParam float re,
            @RequestParam int resolution,
            @RequestParam(defaultValue = "1.0") float ub,
            @RequestParam(defaultValue = "0.3") float cfl) {
        try {
            SimulationConfig cfg = engineService.getCurrentConfig();
            float l = 1.0f;
            float dx = l / resolution;
            cfg.setNx(resolution);
            cfg.setNy(resolution);
            cfg.setNz(resolution);
            cfg.setDx(dx);
            cfg.setDy(dx);
            cfg.setDz(dx);
            cfg.setUb(ub);
            cfg.setNu(ub * l / re);
            cfg.setDt(cfl * dx / ub);
            cfg.getBcZmax().setU(ub);
            engineService.initializeSolver(cfg);
            engineService.publishCurrentSlice();
            // Resolution changed -- any previously generated mesh (voxelized at the
            // old nx/ny/nz) is now stale and must be regenerated before Run.
            meshUpToDate = false;
            cadService.regeneratePreview(cfg.getNx(), cfg.getNy(), cfg.getNz(),
                    cfg.getNx() * cfg.getDx(), cfg.getNy() * cfg.getDy(), cfg.getNz() * cfg.getDz());
            float actualCfl = cfg.getUb() * cfg.getDt() / cfg.getDx();
            return ResponseEntity.ok("{\"status\":\"applied\",\"re\":" + re + ",\"resolution\":" + resolution
                    + ",\"nu\":" + cfg.getNu() + ",\"dt\":" + cfg.getDt() + ",\"cfl\":" + actualCfl + "}");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }
}
