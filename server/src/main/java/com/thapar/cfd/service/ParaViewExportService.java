package com.thapar.cfd.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

@Service
public class ParaViewExportService {
    private static final Logger log = LoggerFactory.getLogger(ParaViewExportService.class);
    private static final String WORKSPACE_ROOT = "/workspace/thapar CFD platform";
    private static final String BRIDGE_SCRIPT = WORKSPACE_ROOT + "/tools/paraview_export_bridge.py";
    private static final String EXPORT_DIR = WORKSPACE_ROOT + "/server/target/exports";
    private static final String LIVE_FIELDS_DIR = EXPORT_DIR + "/live_fields";

    private final CfdEngineService engineService;

    public ParaViewExportService(CfdEngineService engineService) {
        this.engineService = engineService;
        new File(EXPORT_DIR).mkdirs();
    }

    /** Dumps the live solver's real 3D fields to disk so every bridge call below
     *  operates on actual simulation state rather than a synthetic placeholder. */
    private File refreshLiveFields() {
        return engineService.dumpFieldsForExport(new File(LIVE_FIELDS_DIR));
    }

    public File exportVti(int nx, int ny, int nz, float dx, float dy, float dz) throws IOException, InterruptedException {
        File fieldsDir = refreshLiveFields();
        String vtiPath = EXPORT_DIR + "/flowstudio_solution.vti";
        ProcessBuilder pb = new ProcessBuilder(
            "python3", BRIDGE_SCRIPT,
            "--nx", String.valueOf(nx),
            "--ny", String.valueOf(ny),
            "--nz", String.valueOf(nz),
            "--dx", String.valueOf(dx),
            "--dy", String.valueOf(dy),
            "--dz", String.valueOf(dz),
            "--fields-dir", fieldsDir.getAbsolutePath(),
            "--out-vti", vtiPath
        );
        pb.redirectErrorStream(true);
        Process p = pb.start();
        int exitCode = p.waitFor();

        if (exitCode != 0) {
            String err = new String(p.getInputStream().readAllBytes());
            log.error("ParaView export failed: {}", err);
            throw new RuntimeException("Export failed: " + err);
        }

        log.info("Exported binary VTK ImageData to {}", vtiPath);
        return new File(vtiPath);
    }

    public String generateStreamlines(int nx, int ny, int nz, float dx, float dy, float dz) throws IOException, InterruptedException {
        File fieldsDir = refreshLiveFields();
        String jsonPath = EXPORT_DIR + "/streamlines.json";
        ProcessBuilder pb = new ProcessBuilder(
            "python3", BRIDGE_SCRIPT,
            "--nx", String.valueOf(nx),
            "--ny", String.valueOf(ny),
            "--nz", String.valueOf(nz),
            "--dx", String.valueOf(dx),
            "--dy", String.valueOf(dy),
            "--dz", String.valueOf(dz),
            "--fields-dir", fieldsDir.getAbsolutePath(),
            "--out-streamlines", jsonPath
        );
        pb.redirectErrorStream(true);
        Process p = pb.start();
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            String err = new String(p.getInputStream().readAllBytes());
            log.error("Streamlines generation failed: {}", err);
            return "{\"num_lines\":0,\"lines\":[]}";
        }
        return java.nio.file.Files.readString(new File(jsonPath).toPath());
    }

    public String generateIsosurface(int nx, int ny, int nz, float dx, float dy, float dz, float isovalue) throws IOException, InterruptedException {
        File fieldsDir = refreshLiveFields();
        String jsonPath = EXPORT_DIR + "/isosurface.json";
        ProcessBuilder pb = new ProcessBuilder(
            "python3", BRIDGE_SCRIPT,
            "--nx", String.valueOf(nx),
            "--ny", String.valueOf(ny),
            "--nz", String.valueOf(nz),
            "--dx", String.valueOf(dx),
            "--dy", String.valueOf(dy),
            "--dz", String.valueOf(dz),
            "--fields-dir", fieldsDir.getAbsolutePath(),
            "--isovalue", String.valueOf(isovalue),
            "--out-isosurface", jsonPath
        );
        pb.redirectErrorStream(true);
        Process p = pb.start();
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            String err = new String(p.getInputStream().readAllBytes());
            log.error("Isosurface generation failed: {}", err);
            return "{\"num_vertices\":0,\"num_faces\":0,\"vertices\":[],\"faces\":[]}";
        }
        return java.nio.file.Files.readString(new File(jsonPath).toPath());
    }

    public String probePoint(float x, float y, float z, int nx, int ny, int nz, float dx, float dy, float dz) throws IOException, InterruptedException {
        File fieldsDir = refreshLiveFields();
        ProcessBuilder pb = new ProcessBuilder(
            "python3", BRIDGE_SCRIPT,
            "--nx", String.valueOf(nx),
            "--ny", String.valueOf(ny),
            "--nz", String.valueOf(nz),
            "--dx", String.valueOf(dx),
            "--dy", String.valueOf(dy),
            "--dz", String.valueOf(dz),
            "--fields-dir", fieldsDir.getAbsolutePath(),
            "--probe-x", String.valueOf(x),
            "--probe-y", String.valueOf(y),
            "--probe-z", String.valueOf(z)
        );
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String output = new String(p.getInputStream().readAllBytes()).trim();
        p.waitFor();
        return output;
    }
}
