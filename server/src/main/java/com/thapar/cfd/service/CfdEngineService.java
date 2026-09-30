package com.thapar.cfd.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.thapar.cfd.model.SimulationConfig;
import com.thapar.cfd.model.SolverDiagnostics;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

@Service
public class CfdEngineService {
    private static final Logger log = LoggerFactory.getLogger(CfdEngineService.class);
    private static final String LIB_PATH = "/workspace/thapar CFD platform/src/cuda/build/libthaparcfd.so";

    private final Linker linker;
    private final SymbolLookup lookup;
    private final Arena globalArena;

    private MethodHandle createFromJson;
    private MethodHandle stepMultiple;
    private MethodHandle getSliceHandle;
    private MethodHandle getField3dHandle;
    private MethodHandle getDiagnosticsHandle;
    private MethodHandle setObstacleMaskHandle;
    private MethodHandle destroySolverHandle;

    private MemorySegment solverPtr = MemorySegment.NULL;
    private SimulationConfig currentConfig = new SimulationConfig();
    private final SolverDiagnostics diagnostics = new SolverDiagnostics();

    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private ExecutorService simExecutor;
    private SliceConsumer sliceConsumer;

    // Bounded "Run" state (distinct from unbounded "Live Preview"): -1 target
    // means unbounded, checkpointIntervalSteps 0 means no periodic snapshots.
    private volatile long targetStep = -1;
    private volatile long checkpointIntervalSteps = 0;
    private volatile CheckpointCallback checkpointCallback;

    public interface SliceConsumer {
        void onSliceAvailable(ByteBuffer buffer, int width, int height, int axisId, String fieldName, float minVal, float maxVal);
    }

    public interface CheckpointCallback {
        void onCheckpointDue(long step, double physicalTime);
    }

    public void setCheckpointCallback(CheckpointCallback cb) {
        this.checkpointCallback = cb;
    }

    public CfdEngineService() {
        this.linker = Linker.nativeLinker();
        this.globalArena = Arena.ofShared();
        this.lookup = SymbolLookup.libraryLookup(Path.of(LIB_PATH), globalArena);
        initMethodHandles();
    }

    private void initMethodHandles() {
        try {
            createFromJson = linker.downcallHandle(
                lookup.find("thapar_create_solver_from_json").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );
            stepMultiple = linker.downcallHandle(
                lookup.find("thapar_step_multiple").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );
            getSliceHandle = linker.downcallHandle(
                lookup.find("thapar_get_slice").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );
            getField3dHandle = linker.downcallHandle(
                lookup.find("thapar_get_field_3d").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );
            getDiagnosticsHandle = linker.downcallHandle(
                lookup.find("thapar_get_diagnostics").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );
            setObstacleMaskHandle = linker.downcallHandle(
                lookup.find("thapar_set_obstacle_mask").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );
            destroySolverHandle = linker.downcallHandle(
                lookup.find("thapar_destroy_solver").orElseThrow(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );
            log.info("Initialized Java 21 Panama FFM handles to libthaparcfd.so");
        } catch (Exception e) {
            log.error("Failed to link CUDA methods via Panama FFM", e);
            throw new RuntimeException(e);
        }
    }

    public synchronized void initializeSolver(SimulationConfig config) {
        destroySolver();
        this.currentConfig = config;
        try {
            initializeSolverFromRawJson(buildEngineJson(config));
            log.info("Created native CUDA solver instance: {} cells", config.getNx() * config.getNy() * config.getNz());
        } catch (Throwable t) {
            log.error("Failed to initialize solver", t);
            throw new RuntimeException(t);
        }
    }

    /** Builds the flat sim_spec.json IR consumed by thapar_create_solver_from_json. */
    public static String buildEngineJson(SimulationConfig config) {
        StringBuilder json = new StringBuilder("{");
        json.append("\"nx\":").append(config.getNx()).append(",");
        json.append("\"ny\":").append(config.getNy()).append(",");
        json.append("\"nz\":").append(config.getNz()).append(",");
        json.append("\"dx\":").append(config.getDx()).append(",");
        json.append("\"dy\":").append(config.getDy()).append(",");
        json.append("\"dz\":").append(config.getDz()).append(",");
        json.append("\"dt\":").append(config.getDt()).append(",");
        json.append("\"nu\":").append(config.getNu()).append(",");
        json.append("\"rho\":").append(config.getRho()).append(",");
        json.append("\"ub\":").append(config.getUb()).append(",");
        json.append("\"nlevel\":").append(config.getNlevel()).append(",");
        json.append("\"mg_iterations\":").append(config.getMgIterations()).append(",");
        json.append("\"enable_heat\":").append(config.isEnableHeat() ? 1 : 0).append(",");
        json.append("\"enable_turbulence\":").append(config.isEnableTurbulence() ? 1 : 0).append(",");
        json.append("\"enable_vof\":").append(config.isEnableVof() ? 1 : 0).append(",");
        json.append("\"thermal_diffusivity\":").append(config.getThermalDiffusivity()).append(",");
        json.append("\"beta_thermal\":").append(config.getBetaThermal()).append(",");
        json.append("\"t_ref\":").append(config.gettRef()).append(",");
        json.append("\"gx\":").append(config.getGx()).append(",");
        json.append("\"gy\":").append(config.getGy()).append(",");
        json.append("\"gz\":").append(config.getGz()).append(",");
        json.append("\"rho1\":").append(config.getRho1()).append(",");
        json.append("\"rho2\":").append(config.getRho2()).append(",");
        json.append("\"nu1\":").append(config.getNu1()).append(",");
        json.append("\"nu2\":").append(config.getNu2()).append(",");
        json.append("\"c_alpha\":").append(config.getcAlpha()).append(",");
        appendBoundaryPatch(json, "xmin", config.getBcXmin());
        appendBoundaryPatch(json, "xmax", config.getBcXmax());
        appendBoundaryPatch(json, "ymin", config.getBcYmin());
        appendBoundaryPatch(json, "ymax", config.getBcYmax());
        appendBoundaryPatch(json, "zmin", config.getBcZmin());
        appendBoundaryPatch(json, "zmax", config.getBcZmax());
        // strip trailing comma left by the last appendBoundaryPatch call
        json.setLength(json.length() - 1);
        json.append("}");
        return json.toString();
    }

    /** Applies an Expert-mode-edited IR string: rebuilds the native solver AND
     *  best-effort syncs the top-level scalar fields back into currentConfig
     *  so the structured panels (Student wizard, BC picker) stay consistent. */
    public synchronized void applyRawJsonConfig(String json) {
        pauseSimulation();
        try {
            java.util.Map<String, Object> parsed = new ObjectMapper().readValue(json, java.util.Map.class);
            SimulationConfig cfg = this.currentConfig;
            setIfPresent(parsed, "nx", v -> cfg.setNx(((Number) v).intValue()));
            setIfPresent(parsed, "ny", v -> cfg.setNy(((Number) v).intValue()));
            setIfPresent(parsed, "nz", v -> cfg.setNz(((Number) v).intValue()));
            setIfPresent(parsed, "dx", v -> cfg.setDx(((Number) v).floatValue()));
            setIfPresent(parsed, "dy", v -> cfg.setDy(((Number) v).floatValue()));
            setIfPresent(parsed, "dz", v -> cfg.setDz(((Number) v).floatValue()));
            setIfPresent(parsed, "dt", v -> cfg.setDt(((Number) v).floatValue()));
            setIfPresent(parsed, "nu", v -> cfg.setNu(((Number) v).floatValue()));
            setIfPresent(parsed, "rho", v -> cfg.setRho(((Number) v).floatValue()));
            setIfPresent(parsed, "ub", v -> cfg.setUb(((Number) v).floatValue()));
            setIfPresent(parsed, "nlevel", v -> cfg.setNlevel(((Number) v).intValue()));
            setIfPresent(parsed, "mg_iterations", v -> cfg.setMgIterations(((Number) v).intValue()));
            setIfPresent(parsed, "enable_heat", v -> cfg.setEnableHeat(truthy(v)));
            setIfPresent(parsed, "enable_turbulence", v -> cfg.setEnableTurbulence(truthy(v)));
            setIfPresent(parsed, "enable_vof", v -> cfg.setEnableVof(truthy(v)));
            setIfPresent(parsed, "thermal_diffusivity", v -> cfg.setThermalDiffusivity(((Number) v).floatValue()));
            setIfPresent(parsed, "beta_thermal", v -> cfg.setBetaThermal(((Number) v).floatValue()));
            setIfPresent(parsed, "t_ref", v -> cfg.settRef(((Number) v).floatValue()));
            setIfPresent(parsed, "gx", v -> cfg.setGx(((Number) v).floatValue()));
            setIfPresent(parsed, "gy", v -> cfg.setGy(((Number) v).floatValue()));
            setIfPresent(parsed, "gz", v -> cfg.setGz(((Number) v).floatValue()));
            setIfPresent(parsed, "rho1", v -> cfg.setRho1(((Number) v).floatValue()));
            setIfPresent(parsed, "rho2", v -> cfg.setRho2(((Number) v).floatValue()));
            setIfPresent(parsed, "nu1", v -> cfg.setNu1(((Number) v).floatValue()));
            setIfPresent(parsed, "nu2", v -> cfg.setNu2(((Number) v).floatValue()));
            initializeSolver(cfg);
            publishCurrentSlice();
        } catch (Exception e) {
            log.error("Failed to apply Expert-mode IR JSON", e);
            throw new RuntimeException("Invalid sim_spec.json: " + e.getMessage(), e);
        }
    }

    private static boolean truthy(Object v) {
        if (v instanceof Boolean b) return b;
        if (v instanceof Number n) return n.intValue() != 0;
        return Boolean.parseBoolean(String.valueOf(v));
    }

    private static void setIfPresent(java.util.Map<String, Object> map, String key, java.util.function.Consumer<Object> setter) {
        if (map.containsKey(key) && map.get(key) != null) setter.accept(map.get(key));
    }

    public synchronized void applyBoundaryCondition(String face, String type, float u, float v, float w) {
        SimulationConfig.BoundaryPatch patch = currentConfig.getBoundaryPatch(face);
        patch.setType(type);
        patch.setU(u);
        patch.setV(v);
        patch.setW(w);
        initializeSolver(currentConfig);
        publishCurrentSlice();
    }

    private static void appendBoundaryPatch(StringBuilder json, String face, SimulationConfig.BoundaryPatch bc) {
        json.append("\"bc_").append(face).append("_type\":\"").append(bc.getType()).append("\",");
        json.append("\"bc_").append(face).append("_u\":").append(bc.getU()).append(",");
        json.append("\"bc_").append(face).append("_v\":").append(bc.getV()).append(",");
        json.append("\"bc_").append(face).append("_w\":").append(bc.getW()).append(",");
        json.append("\"bc_").append(face).append("_p\":").append(bc.getP()).append(",");
        json.append("\"bc_").append(face).append("_t\":").append(bc.getT()).append(",");
    }

    /** Sends a raw sim_spec.json IR string straight to the native engine (Expert-mode direct IR editing). */
    public synchronized void initializeSolverFromRawJson(String json) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment jsonSeg = arena.allocateUtf8String(json);
            this.solverPtr = (MemorySegment) createFromJson.invoke(jsonSeg);
            diagnostics.setCurrentStep(0);
            diagnostics.setPhysicalTime(0.0);
        } catch (Throwable t) {
            log.error("Failed to initialize solver from raw JSON", t);
            throw new RuntimeException(t);
        }
    }

    /** Registers a passive slice consumer (e.g. the WebSocket stream handler) WITHOUT
     *  starting the compute loop. Safe to call at application startup: it neither
     *  allocates a solver nor consumes any GPU cycles until a user explicitly starts
     *  a run (RUN button / "Start Live Preview") -- see startContinuousSimulation(). */
    public synchronized void registerSliceConsumer(SliceConsumer consumer) {
        this.sliceConsumer = consumer;
    }

    /** Explicit, user-initiated opt-in: allocates the solver if needed and starts the
     *  continuous GPU compute loop. Must only ever be invoked in response to a direct
     *  user action (RUN / "Start Live Preview") -- never from application startup.
     *  Does NOT touch the registered slice consumer (see registerSliceConsumer) so
     *  starting/stopping runs never disturbs the WebSocket stream registration. */
    public synchronized void startContinuousSimulation() {
        if (solverPtr.equals(MemorySegment.NULL)) {
            initializeSolver(currentConfig);
        }
        this.targetStep = -1; // unbounded
        this.checkpointIntervalSteps = 0;
        diagnostics.setTargetStep(-1);
        diagnostics.setRunCompleted(false);
        if (isRunning.compareAndSet(false, true)) {
            diagnostics.setRunning(true);
            simExecutor = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "CFD-Compute-Worker");
                t.setDaemon(true);
                return t;
            });

            simExecutor.submit(this::simulationLoop);
            log.info("Simulation continuous loop started");
        }
    }

    /** Explicit, user-initiated opt-in (same GPU-consent rule as
     *  startContinuousSimulation): runs exactly `steps` more steps from the
     *  current position, capturing a checkpoint every `checkpointIntervalSteps`
     *  (0 disables checkpointing), then auto-stops and marks
     *  diagnostics.runCompleted -- the "actual run, come back to see results"
     *  mode, distinct from the unbounded, manually-stopped Live Preview. */
    public synchronized void startRun(long steps, long checkpointIntervalSteps) {
        if (solverPtr.equals(MemorySegment.NULL)) {
            initializeSolver(currentConfig);
        }
        this.targetStep = diagnostics.getCurrentStep() + Math.max(steps, 1);
        this.checkpointIntervalSteps = Math.max(checkpointIntervalSteps, 0);
        diagnostics.setTargetStep(this.targetStep);
        diagnostics.setRunCompleted(false);
        if (isRunning.compareAndSet(false, true)) {
            diagnostics.setRunning(true);
            simExecutor = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "CFD-Compute-Worker");
                t.setDaemon(true);
                return t;
            });
            simExecutor.submit(this::simulationLoop);
            log.info("Bounded run started: target step {}, checkpoint every {} steps", targetStep, this.checkpointIntervalSteps);
        }
    }

    public synchronized void pauseSimulation() {
        if (isRunning.compareAndSet(true, false)) {
            diagnostics.setRunning(false);
            if (simExecutor != null) {
                simExecutor.shutdown();
                try {
                    if (!simExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                        simExecutor.shutdownNow();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                simExecutor = null;
            }
            log.info("Simulation paused at step {}", diagnostics.getCurrentStep());
        }
    }

    public synchronized void singleStep(int steps) {
        if (solverPtr.equals(MemorySegment.NULL)) {
            initializeSolver(currentConfig);
        }
        try {
            stepMultiple.invoke(solverPtr, steps);
            diagnostics.setCurrentStep(diagnostics.getCurrentStep() + steps);
            diagnostics.setPhysicalTime(diagnostics.getCurrentStep() * currentConfig.getDt());
            updateDiagnosticsInternal();
            publishCurrentSlice();
        } catch (Throwable t) {
            log.error("Single step failed", t);
        }
    }

    public synchronized void resetSimulation() {
        pauseSimulation();
        initializeSolver(currentConfig);
        publishCurrentSlice();
    }

    public synchronized void setObstacleMask(byte[] floatBytes) {
        if (solverPtr.equals(MemorySegment.NULL)) {
            initializeSolver(currentConfig);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(floatBytes.length);
            MemorySegment.copy(floatBytes, 0, seg, ValueLayout.JAVA_BYTE, 0, floatBytes.length);
            int count = floatBytes.length / Float.BYTES;
            setObstacleMaskHandle.invoke(solverPtr, seg, (long) count);
            log.info("Uploaded obstacle mask ({} cells) to CUDA solver", count);
        } catch (Throwable t) {
            log.error("Failed to set obstacle mask", t);
        }
    }

    private void simulationLoop() {
        int stepsPerBatch = 5;
        long lastTime = System.nanoTime();
        long frameCount = 0;
        long lastCheckpointStep = diagnostics.getCurrentStep();

        while (isRunning.get() && !Thread.currentThread().isInterrupted()) {
            try {
                stepMultiple.invoke(solverPtr, stepsPerBatch);
                long currentStep = diagnostics.getCurrentStep() + stepsPerBatch;
                diagnostics.setCurrentStep(currentStep);
                diagnostics.setPhysicalTime(currentStep * currentConfig.getDt());
                frameCount += stepsPerBatch;

                long now = System.nanoTime();
                if (now - lastTime >= 500_000_000L) { // Update stats every 500ms
                    double fps = (frameCount * 1e9) / (now - lastTime);
                    diagnostics.setFramesPerSecond(fps);
                    updateDiagnosticsInternal();
                    lastTime = now;
                    frameCount = 0;
                }

                boolean reachedTarget = (targetStep >= 0 && currentStep >= targetStep);

                if (checkpointIntervalSteps > 0 && (currentStep - lastCheckpointStep >= checkpointIntervalSteps || reachedTarget)) {
                    lastCheckpointStep = currentStep;
                    updateDiagnosticsInternal();
                    CheckpointCallback cb = checkpointCallback;
                    if (cb != null) {
                        try {
                            cb.onCheckpointDue(currentStep, diagnostics.getPhysicalTime());
                        } catch (Throwable t) {
                            log.error("Checkpoint capture failed at step {}", currentStep, t);
                        }
                    }
                }

                publishCurrentSlice();

                if (reachedTarget) {
                    updateDiagnosticsInternal();
                    diagnostics.setRunning(false);
                    diagnostics.setRunCompleted(true);
                    isRunning.set(false);
                    log.info("Bounded run reached target step {} -- auto-stopped", currentStep);
                    break;
                }

                Thread.sleep(1); // Yield to prevent bus starvation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                log.error("Simulation error in loop", t);
                pauseSimulation();
                break;
            }
        }
    }

    private void updateDiagnosticsInternal() {
        if (solverPtr.equals(MemorySegment.NULL)) return;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment maxDiv = arena.allocate(ValueLayout.JAVA_FLOAT);
            MemorySegment pRes = arena.allocate(ValueLayout.JAVA_FLOAT);
            MemorySegment massErr = arena.allocate(ValueLayout.JAVA_FLOAT);
            MemorySegment vram = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment lastMs = arena.allocate(ValueLayout.JAVA_DOUBLE);

            getDiagnosticsHandle.invoke(solverPtr, maxDiv, pRes, massErr, vram, lastMs);
            diagnostics.setMaxDivergence(maxDiv.get(ValueLayout.JAVA_FLOAT, 0));
            diagnostics.setPressureResidual(pRes.get(ValueLayout.JAVA_FLOAT, 0));
            diagnostics.setMassFluxError(massErr.get(ValueLayout.JAVA_FLOAT, 0));
            diagnostics.setVramBytes(vram.get(ValueLayout.JAVA_LONG, 0));
            diagnostics.setLastStepMs(lastMs.get(ValueLayout.JAVA_DOUBLE, 0));
        } catch (Throwable t) {
            log.error("Failed to read diagnostics", t);
        }
    }

    /** Streams every currently-enabled cut-plane (simultaneous axial/coronal/sagittal
     *  ParaView-style slicing) in one pass; each plane is tagged with its axis id so
     *  the client can route it to the right Three.js plane texture. */
    public synchronized void publishCurrentSlice() {
        if (sliceConsumer == null || solverPtr.equals(MemorySegment.NULL)) return;

        int fieldId = switch (currentConfig.getActiveField().toUpperCase()) {
            case "PRESSURE" -> 4;
            case "TEMPERATURE" -> 5;
            case "TURBULENCE" -> 6;
            case "VOF_ALPHA" -> 7;
            default -> 3; // VELOCITY_MAGNITUDE
        };

        if (currentConfig.isShowSliceX()) publishAxisSlice(fieldId, 0, currentConfig.getSliceIndexX());
        if (currentConfig.isShowSliceY()) publishAxisSlice(fieldId, 1, currentConfig.getSliceIndexY());
        if (currentConfig.isShowSliceZ()) publishAxisSlice(fieldId, 2, currentConfig.getSliceIndexZ());
    }

    private void publishAxisSlice(int fieldId, int axisId, int sliceIndex) {
        int w, h;
        if (axisId == 0) { w = currentConfig.getNy(); h = currentConfig.getNz(); }
        else if (axisId == 1) { w = currentConfig.getNx(); h = currentConfig.getNz(); }
        else { w = currentConfig.getNx(); h = currentConfig.getNy(); }

        int totalElements = w * h;
        long byteSize = (long) totalElements * Float.BYTES;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outBuf = arena.allocate(byteSize);
            int status = (int) getSliceHandle.invoke(solverPtr, fieldId, axisId, sliceIndex, outBuf, byteSize);
            if (status != 0) return;

            ByteBuffer bb = outBuf.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            float minVal = Float.MAX_VALUE;
            float maxVal = -Float.MAX_VALUE;
            for (int i = 0; i < totalElements; i++) {
                float val = bb.getFloat(i * Float.BYTES);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            sliceConsumer.onSliceAvailable(bb, w, h, axisId, currentConfig.getActiveField(), minVal, maxVal);
        } catch (Throwable t) {
            log.error("Error extracting slice", t);
        }
    }

    /** Field ids matching thapar_field_type_t, keyed by the raw filename the
     *  ParaView bridge (tools/paraview_export_bridge.py --fields-dir) expects. */
    private static final java.util.Map<String, Integer> EXPORT_FIELD_IDS = java.util.Map.of(
        "u", 0, "v", 1, "w", 2, "pressure", 4, "temperature", 5, "nu_t", 6, "alpha", 7
    );

    /** Dumps the live solver's current 3D fields to raw float32 binary files (shape
     *  nz*ny*nx, row-major, native byte order) so the headless ParaView/VTK bridge can
     *  compute real streamlines, isosurfaces, VTI exports, and point probes against
     *  actual simulation state instead of a synthetic placeholder field. */
    public synchronized File dumpFieldsForExport(File dir) {
        if (solverPtr.equals(MemorySegment.NULL)) return dir;
        dir.mkdirs();
        long count = (long) currentConfig.getNx() * currentConfig.getNy() * currentConfig.getNz();
        long byteSize = count * Float.BYTES;

        java.util.Set<String> active = new java.util.HashSet<>(java.util.List.of("u", "v", "w", "pressure"));
        if (currentConfig.isEnableHeat()) active.add("temperature");
        if (currentConfig.isEnableTurbulence()) active.add("nu_t");
        if (currentConfig.isEnableVof()) active.add("alpha");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outBuf = arena.allocate(byteSize);
            for (String name : active) {
                int fieldId = EXPORT_FIELD_IDS.get(name);
                int status = (int) getField3dHandle.invoke(solverPtr, fieldId, outBuf, byteSize);
                if (status != 0) continue;
                try (var channel = java.nio.channels.FileChannel.open(new File(dir, name + ".bin").toPath(),
                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.WRITE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING)) {
                    channel.write(outBuf.asByteBuffer());
                }
            }
        } catch (Throwable t) {
            log.error("Failed to dump fields for export", t);
        }
        return dir;
    }

    public SolverDiagnostics getDiagnostics() {
        return diagnostics;
    }

    public SimulationConfig getCurrentConfig() {
        return currentConfig;
    }

    @PreDestroy
    public synchronized void destroySolver() {
        pauseSimulation();
        try {
            Thread.sleep(50);
        } catch (InterruptedException ignored) {}
        if (!solverPtr.equals(MemorySegment.NULL)) {
            try {
                destroySolverHandle.invoke(solverPtr);
                log.info("Destroyed native solver instance");
            } catch (Throwable t) {
                log.warn("Error destroying solver", t);
            }
            solverPtr = MemorySegment.NULL;
        }
    }
}
