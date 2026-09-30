package com.thapar.cfd.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Multi-body CAD scene manager. Real parametric CAD -- not a single uploaded
 * mesh -- backed by tools/cad_kernel_freecad.py, which runs under REAL
 * FreeCAD 1.1.3 (installed via conda-forge into the `cad` micromamba
 * environment at /opt/conda_envs/envs/cad -- `import FreeCAD` genuinely
 * works there; earlier revisions of this file drove OpenCASCADE indirectly
 * through gmsh's bindings because building FreeCAD's own C++/Qt application
 * from the source checked out in reference/freecad was ruled impractical.
 * The conda-forge package sidesteps that: it ships FreeCAD already compiled).
 *
 * Every geometric operation is FreeCAD's actual Part API: Part.makeBox/
 * makeCylinder/makeSphere/makeCone/makeTorus, Face.extrude/Wire.revolve for
 * sketches, shape.fuse/.cut/.common for boolean ops, shape.makeFillet/
 * .makeChamfer for real edge rounding, Part.read/shape.exportStep for true
 * B-Rep STEP/IGES I/O. See tools/cad_kernel_freecad.py's header for the two
 * real FreeCAD quirks worked around there (freecadcmd runs scripts as an
 * imported module, not "__main__"; it does not propagate the parent shell's
 * environment into that module, so job parameters are handed off via a
 * fixed file path rather than argv or an env var).
 *
 * Bodies are addressable, individually editable, and combinable via boolean
 * operations. Editing the scene regenerates the preview mesh + Brinkman
 * voxelization mask on disk immediately (cheap, client-facing), but does NOT
 * push anything to the GPU solver -- that only happens when the Mesh stage's
 * "Generate Mesh" action explicitly commits the current mask
 * (see CfdEngineService#setObstacleMask), keeping geometry editing and GPU
 * resource use clearly separated.
 */
@Service
public class CadPreprocessService {
    private static final Logger log = LoggerFactory.getLogger(CadPreprocessService.class);
    private static final String WORKSPACE_ROOT = "/workspace/thapar CFD platform";
    private static final String KERNEL_SCRIPT = WORKSPACE_ROOT + "/tools/cad_kernel_freecad.py";
    private static final String CAD_CACHE_DIR = WORKSPACE_ROOT + "/server/target/cad_cache";
    private static final String UPLOAD_DIR = CAD_CACHE_DIR + "/uploads";
    // Calling freecadcmd directly (confirmed self-contained via rpath -- runs
    // correctly even with `env -i`, a fully stripped environment) rather than
    // through `micromamba run -n cad freecadcmd ...`: the wrapper re-activates
    // the whole conda environment on every single invocation, which is pure
    // overhead the binary itself doesn't need.
    private static final String FREECADCMD_BIN = "/opt/conda_envs/envs/cad/bin/freecadcmd";
    private static final String JOB_PATH = CAD_CACHE_DIR + "/freecad_job.json";

    private final ObjectMapper mapper = new ObjectMapper();
    private final List<ObjectNode> bodies = Collections.synchronizedList(new ArrayList<>());
    private final AtomicInteger idCounter = new AtomicInteger(1);

    private volatile String lastGeometryJson = "{\"bodies\":[],\"num_bodies\":0}";
    private volatile String lastError = null;

    public CadPreprocessService() {
        new File(CAD_CACHE_DIR).mkdirs();
        new File(UPLOAD_DIR).mkdirs();
    }

    public record RegenResult(String geometryJson, String maskPath, boolean ok, String error) {}

    public synchronized String getSceneJson() {
        ArrayNode arr = mapper.createArrayNode();
        bodies.forEach(arr::add);
        ObjectNode root = mapper.createObjectNode();
        root.set("bodies", arr);
        root.put("count", bodies.size());
        return root.toString();
    }

    public synchronized String getLastGeometryJson() {
        return lastGeometryJson;
    }

    public synchronized String addBody(String type, JsonNode params, JsonNode transform, String role, String label) {
        String id = "body" + idCounter.getAndIncrement();
        ObjectNode node = mapper.createObjectNode();
        node.put("id", id);
        node.put("type", type);
        node.put("label", (label == null || label.isBlank()) ? (type + " " + id) : label);
        node.put("role", (role == null) ? "obstacle" : role);
        node.set("params", params);
        node.set("transform", transform != null ? transform : mapper.createObjectNode());
        bodies.add(node);
        return id;
    }

    public synchronized boolean updateBody(String id, JsonNode params, JsonNode transform) {
        for (ObjectNode b : bodies) {
            if (b.get("id").asText().equals(id)) {
                if (params != null) b.set("params", params);
                if (transform != null) b.set("transform", transform);
                return true;
            }
        }
        return false;
    }

    public synchronized boolean deleteBody(String id) {
        // Refuse to delete a body that a boolean/fillet/chamfer operation still
        // references, so the scene never silently breaks (error prevention).
        for (ObjectNode b : bodies) {
            if (b.get("id").asText().equals(id)) continue;
            String type = b.get("type").asText();
            JsonNode p = b.get("params");
            if ("boolean".equals(type) && (p.get("a").asText().equals(id) || p.get("b").asText().equals(id))) return false;
            if (("fillet".equals(type) || "chamfer".equals(type)) && p.get("body").asText().equals(id)) return false;
        }
        return bodies.removeIf(b -> b.get("id").asText().equals(id));
    }

    public synchronized String addBoolean(String aId, String bId, String op, String label) {
        ObjectNode params = mapper.createObjectNode();
        params.put("a", aId);
        params.put("b", bId);
        params.put("op", op);
        return addBody("boolean", params, null, "obstacle", label);
    }

    public synchronized String importFile(File uploadedFile, String label) {
        ObjectNode params = mapper.createObjectNode();
        params.put("file_path", uploadedFile.getAbsolutePath());
        return addBody("imported", params, null, "obstacle", label != null ? label : uploadedFile.getName());
    }

    public String getUploadDir() {
        return UPLOAD_DIR;
    }

    public synchronized boolean hasBody(String id) {
        return bodies.stream().anyMatch(b -> b.get("id").asText().equals(id));
    }

    public synchronized void clearScene() {
        bodies.clear();
    }

    private ObjectNode buildSceneNode() {
        ObjectNode root = mapper.createObjectNode();
        ArrayNode arr = mapper.createArrayNode();
        bodies.forEach(arr::add);
        root.set("bodies", arr);
        return root;
    }

    /** Runs tools/cad_kernel_freecad.py under real FreeCAD (freecadcmd, via the
     *  `cad` micromamba env) with the given job parameters written to the fixed
     *  job-file path freecadcmd's script loader actually reads from. */
    private String runFreecadKernel(int nx, int ny, int nz, float lx, float ly, float lz,
                                     String exportStepPath, String exportStlPath) throws IOException, InterruptedException {
        String jsonPath = CAD_CACHE_DIR + "/cad_geometry.json";
        String maskPath = CAD_CACHE_DIR + "/obstacle_mask.bin";

        ObjectNode job = mapper.createObjectNode();
        job.set("scene", buildSceneNode());
        job.put("nx", nx); job.put("ny", ny); job.put("nz", nz);
        job.put("lx", lx); job.put("ly", ly); job.put("lz", lz);
        job.put("out_mask", maskPath);
        job.put("out_json", jsonPath);
        if (exportStepPath != null) job.put("export_step", exportStepPath);
        if (exportStlPath != null) job.put("export_stl", exportStlPath);
        Files.writeString(Path.of(JOB_PATH), job.toString());

        ProcessBuilder pb = new ProcessBuilder(FREECADCMD_BIN, KERNEL_SCRIPT);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String output = new String(p.getInputStream().readAllBytes());
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException(extractErrorMessage(output));
        }
        return jsonPath;
    }

    /** freecadcmd always prints its own startup banner (version/copyright) to
     *  stdout ahead of anything our script prints, mixed in with our captured
     *  output -- surface just the actual "ERROR: ..." line(s) to the UI
     *  instead of that noise wrapped around a real OCCT/Python exception. */
    private static String extractErrorMessage(String rawOutput) {
        List<String> errorLines = new ArrayList<>();
        for (String line : rawOutput.split("\n")) {
            if (line.contains("ERROR:") || line.contains("Error:")) errorLines.add(line.trim());
        }
        return errorLines.isEmpty() ? rawOutput.trim() : String.join(" | ", errorLines);
    }

    /** Regenerates the Three.js preview JSON and the Brinkman voxelization mask
     *  file on disk for the current scene at the given grid resolution. Does
     *  NOT touch the GPU solver. */
    public synchronized RegenResult regeneratePreview(int nx, int ny, int nz, float lx, float ly, float lz) {
        String maskPath = CAD_CACHE_DIR + "/obstacle_mask.bin";
        try {
            String jsonPath = runFreecadKernel(nx, ny, nz, lx, ly, lz, null, null);
            lastGeometryJson = Files.readString(Path.of(jsonPath));
            lastError = null;
            return new RegenResult(lastGeometryJson, maskPath, true, null);
        } catch (Exception e) {
            log.error("FreeCAD kernel invocation failed", e);
            lastError = e.getMessage();
            return new RegenResult(lastGeometryJson, maskPath, false, e.getMessage());
        }
    }

    public String getLastError() {
        return lastError;
    }

    public byte[] getLatestMaskBytes() throws IOException {
        return Files.readAllBytes(Path.of(CAD_CACHE_DIR + "/obstacle_mask.bin"));
    }

    /** Exports the current scene's combined visible geometry as one triangulated
     *  STL file (mesh interchange -- what most CFD/CAM tools want). */
    public synchronized File exportStl(int nx, int ny, int nz, float lx, float ly, float lz) throws IOException, InterruptedException {
        String path = CAD_CACHE_DIR + "/scene_export.stl";
        runFreecadKernel(nx, ny, nz, lx, ly, lz, null, path);
        return new File(path);
    }

    /** Exports the current scene as a real B-Rep STEP file (true CAD interchange,
     *  not a triangulated approximation) -- via FreeCAD's actual exportStep,
     *  possible because every body is a genuine OCCT solid, not a mesh. */
    public synchronized File exportStep(int nx, int ny, int nz, float lx, float ly, float lz) throws IOException, InterruptedException {
        String path = CAD_CACHE_DIR + "/scene_export.step";
        runFreecadKernel(nx, ny, nz, lx, ly, lz, path, null);
        return new File(path);
    }
}
