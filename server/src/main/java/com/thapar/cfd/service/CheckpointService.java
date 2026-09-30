package com.thapar.cfd.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Saved-state history for the CFD run: periodic field snapshots captured
 * during a bounded "Run" (see CfdEngineService#startRun), independent of the
 * live WebSocket stream. This is what makes "run it and come back to see
 * time-history results" possible instead of only ever watching the live
 * current state -- you can scrub back through what already happened.
 */
@Service
public class CheckpointService {
    private static final Logger log = LoggerFactory.getLogger(CheckpointService.class);
    private static final String WORKSPACE_ROOT = "/workspace/thapar CFD platform";
    private static final String CHECKPOINT_DIR = WORKSPACE_ROOT + "/server/target/checkpoints";

    private final CfdEngineService engineService;
    private final List<CheckpointMeta> checkpoints = new CopyOnWriteArrayList<>();

    public record CheckpointMeta(long step, double physicalTime, long timestampMs, String dir) {}

    public CheckpointService(CfdEngineService engineService) {
        this.engineService = engineService;
        new File(CHECKPOINT_DIR).mkdirs();
    }

    public synchronized CheckpointMeta capture(long step, double physicalTime) {
        String dir = CHECKPOINT_DIR + "/step_" + step;
        File dirFile = new File(dir);
        dirFile.mkdirs();
        engineService.dumpFieldsForExport(dirFile);
        CheckpointMeta meta = new CheckpointMeta(step, physicalTime, System.currentTimeMillis(), dir);
        checkpoints.add(meta);
        log.info("Captured checkpoint at step {} ({} total)", step, checkpoints.size());
        return meta;
    }

    public List<CheckpointMeta> list() {
        return List.copyOf(checkpoints);
    }

    public Optional<CheckpointMeta> get(long step) {
        return checkpoints.stream().filter(c -> c.step() == step).findFirst();
    }

    public synchronized void clear() {
        checkpoints.clear();
        File dir = new File(CHECKPOINT_DIR);
        File[] children = dir.listFiles();
        if (children != null) {
            for (File c : children) deleteRecursive(c);
        }
    }

    private static void deleteRecursive(File f) {
        File[] children = f.listFiles();
        if (children != null) for (File c : children) deleteRecursive(c);
        f.delete();
    }

    /** Extracts a 2D slice from a checkpoint's raw float32 field dump, using the
     *  exact same (z*ny*nx + y*nx + x) indexing convention as the live GPU path
     *  (CfdEngineService#publishAxisSlice) so checkpoint and live views agree. */
    public float[] readSlice(long step, String field, String axis, int index, int nx, int ny, int nz) throws IOException {
        Optional<CheckpointMeta> meta = get(step);
        if (meta.isEmpty()) throw new IOException("No checkpoint at step " + step);
        String dir = meta.get().dir();

        float[] u = null, v = null, w = null, primary = null;
        if ("speed".equals(field)) {
            u = readRaw(dir + "/u.bin", nx * ny * nz);
            v = readRaw(dir + "/v.bin", nx * ny * nz);
            w = readRaw(dir + "/w.bin", nx * ny * nz);
        } else {
            primary = readRaw(dir + "/" + field + ".bin", nx * ny * nz);
        }

        int w2, h2;
        if ("X".equalsIgnoreCase(axis)) { w2 = ny; h2 = nz; }
        else if ("Y".equalsIgnoreCase(axis)) { w2 = nx; h2 = nz; }
        else { w2 = nx; h2 = ny; }

        float[] out = new float[w2 * h2];
        int oi = 0;
        for (int row = 0; row < h2; row++) {
            for (int col = 0; col < w2; col++) {
                int x, y, z;
                if ("X".equalsIgnoreCase(axis)) { x = index; y = col; z = row; }
                else if ("Y".equalsIgnoreCase(axis)) { x = col; y = index; z = row; }
                else { x = col; y = row; z = index; }
                int flat = z * (ny * nx) + y * nx + x;
                if (primary != null) {
                    out[oi++] = primary[flat];
                } else {
                    float uu = u[flat], vv = v[flat], ww = w[flat];
                    out[oi++] = (float) Math.sqrt(uu * uu + vv * vv + ww * ww);
                }
            }
        }
        return out;
    }

    private static float[] readRaw(String path, int expectedCount) throws IOException {
        File f = new File(path);
        if (!f.exists()) throw new IOException("Checkpoint field file missing: " + path);
        try (RandomAccessFile raf = new RandomAccessFile(f, "r"); FileChannel ch = raf.getChannel()) {
            ByteBuffer bb = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(bb);
            bb.flip();
            int count = bb.remaining() / Float.BYTES;
            float[] out = new float[count];
            for (int i = 0; i < count; i++) out[i] = bb.getFloat();
            return out;
        }
    }
}
