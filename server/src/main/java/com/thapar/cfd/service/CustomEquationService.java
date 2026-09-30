package com.thapar.cfd.service;

import com.thapar.cfd.model.SimulationConfig;
import org.springframework.stereotype.Service;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * "Custom equations": lets the user type a spatial expression sigma(x,y,z)
 * that gets evaluated per-cell into a float field and merged with the
 * geometry-derived Brinkman/porosity mask before upload to the native GPU
 * solver via the existing CfdEngineService#setObstacleMask channel.
 *
 * Honest scope: this reuses the solver's existing per-cell penalization
 * field, so it can express arbitrary custom porous regions / spatial forcing
 * masks (e.g. a permeable baffle, a graded resistance zone, a checkerboard
 * pattern). It is NOT a route to an arbitrary new PDE term (e.g. a custom
 * energy source with its own transport) -- the compiled CUDA kernel only
 * consumes this one spatially-varying scalar field, and extending it to a
 * genuinely new equation would require changing that native code, which is
 * out of reach from this service layer.
 */
@Service
public class CustomEquationService {

    public record Result(float[] field, float min, float max, float mean, long nonzeroCells, long totalCells) {}

    public Result evaluateField(String expression, SimulationConfig cfg) {
        int nx = cfg.getNx(), ny = cfg.getNy(), nz = cfg.getNz();
        float dx = cfg.getDx(), dy = cfg.getDy(), dz = cfg.getDz();
        float lx = nx * dx, ly = ny * dy, lz = nz * dz;
        float[] field = new float[nx * ny * nz];
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        double sum = 0;
        long nonzero = 0;
        Map<String, Double> vars = new HashMap<>();
        vars.put("lx", (double) lx);
        vars.put("ly", (double) ly);
        vars.put("lz", (double) lz);

        for (int k = 0; k < nz; k++) {
            float z = (k + 0.5f) * dz;
            for (int j = 0; j < ny; j++) {
                float y = (j + 0.5f) * dy;
                for (int i = 0; i < nx; i++) {
                    float x = (i + 0.5f) * dx;
                    vars.put("x", (double) x);
                    vars.put("y", (double) y);
                    vars.put("z", (double) z);
                    vars.put("xn", lx > 0 ? x / lx : 0.0);
                    vars.put("yn", ly > 0 ? y / ly : 0.0);
                    vars.put("zn", lz > 0 ? z / lz : 0.0);
                    float fv = (float) ExpressionEvaluator.evaluate(expression, vars);
                    int idx = k * (ny * nx) + j * nx + i;
                    field[idx] = fv;
                    if (fv < min) min = fv;
                    if (fv > max) max = fv;
                    sum += fv;
                    if (fv != 0f) nonzero++;
                }
            }
        }
        long total = (long) nx * ny * nz;
        return new Result(field, min, max, (float) (sum / Math.max(total, 1)), nonzero, total);
    }

    /** mode: "add" (default, superimpose onto the existing geometry mask),
     *  "max" (take the stronger of the two at each cell), or "replace" (ignore
     *  the geometry mask entirely and use only the custom field). */
    public byte[] applyToMask(String expression, String mode, SimulationConfig cfg, byte[] baseMaskBytes) {
        Result r = evaluateField(expression, cfg);
        float[] field = r.field();
        float[] base = new float[field.length];
        if (baseMaskBytes != null && baseMaskBytes.length == field.length * Float.BYTES) {
            ByteBuffer bb = ByteBuffer.wrap(baseMaskBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < base.length; i++) base[i] = bb.getFloat();
        }
        float[] merged = new float[field.length];
        for (int i = 0; i < field.length; i++) {
            merged[i] = switch (mode == null ? "add" : mode) {
                case "replace" -> field[i];
                case "max" -> Math.max(base[i], field[i]);
                default -> base[i] + field[i];
            };
        }
        ByteBuffer out = ByteBuffer.allocate(merged.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : merged) out.putFloat(v);
        return out.array();
    }
}
