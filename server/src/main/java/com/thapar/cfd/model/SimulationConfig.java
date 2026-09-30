package com.thapar.cfd.model;

public class SimulationConfig {
    private int nx = 64;
    private int ny = 64;
    private int nz = 16;
    private float dx = 0.015625f;
    private float dy = 0.015625f;
    private float dz = 0.015625f;
    private float dt = 0.001f;
    private float nu = 0.01f;
    private float rho = 1.0f;
    private float ub = 1.0f;
    private int nlevel = 4;
    private int mgIterations = 5;

    // Multiphysics
    private boolean enableHeat = false;
    private boolean enableTurbulence = false;
    private boolean enableVof = false;

    private float thermalDiffusivity = 0.014f;
    private float betaThermal = 1.4f;
    private float tRef = 0.0f;
    private float gx = 0.0f;
    private float gy = 0.0f;
    private float gz = 0.0f;

    private float rho1 = 1000.0f;
    private float rho2 = 1.0f;
    private float nu1 = 1e-6f;
    private float nu2 = 1.5e-5f;
    private float cAlpha = 1.0f;

    // Active Slice Display Settings (legacy single-plane compatibility)
    private String activeField = "VELOCITY"; // VELOCITY, PRESSURE, TEMPERATURE, VOF_ALPHA
    private String sliceAxis = "Z"; // X, Y, Z
    private int sliceIndex = 8;

    // Simultaneous multi cut-plane display (ParaView-style axial/coronal/sagittal),
    // independently toggleable and positionable per axis, all streamed live over
    // the same binary WebSocket (each packet tagged with its axis id).
    private boolean showSliceX = true;
    private boolean showSliceY = true;
    private boolean showSliceZ = true;
    private int sliceIndexX = 32;
    private int sliceIndexY = 32;
    private int sliceIndexZ = 8;

    public boolean isShowSliceX() { return showSliceX; }
    public void setShowSliceX(boolean showSliceX) { this.showSliceX = showSliceX; }
    public boolean isShowSliceY() { return showSliceY; }
    public void setShowSliceY(boolean showSliceY) { this.showSliceY = showSliceY; }
    public boolean isShowSliceZ() { return showSliceZ; }
    public void setShowSliceZ(boolean showSliceZ) { this.showSliceZ = showSliceZ; }
    public int getSliceIndexX() { return sliceIndexX; }
    public void setSliceIndexX(int sliceIndexX) { this.sliceIndexX = sliceIndexX; }
    public int getSliceIndexY() { return sliceIndexY; }
    public void setSliceIndexY(int sliceIndexY) { this.sliceIndexY = sliceIndexY; }
    public int getSliceIndexZ() { return sliceIndexZ; }
    public void setSliceIndexZ(int sliceIndexZ) { this.sliceIndexZ = sliceIndexZ; }

    /** One of the 6 outer-domain boundary patches (xmin/xmax/ymin/ymax/zmin/zmax). */
    public static class BoundaryPatch {
        private String type; // no_slip_wall, free_slip_symmetry, dirichlet_inflow, neumann_outflow, moving_lid, periodic, isothermal_wall
        private float u, v, w, p, t;

        public BoundaryPatch() {}
        public BoundaryPatch(String type, float u, float v, float w) {
            this.type = type; this.u = u; this.v = v; this.w = w;
        }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public float getU() { return u; }
        public void setU(float u) { this.u = u; }
        public float getV() { return v; }
        public void setV(float v) { this.v = v; }
        public float getW() { return w; }
        public void setW(float w) { this.w = w; }
        public float getP() { return p; }
        public void setP(float p) { this.p = p; }
        public float getT() { return t; }
        public void setT(float t) { this.t = t; }
    }

    private BoundaryPatch bcXmin = new BoundaryPatch("no_slip_wall", 0, 0, 0);
    private BoundaryPatch bcXmax = new BoundaryPatch("no_slip_wall", 0, 0, 0);
    private BoundaryPatch bcYmin = new BoundaryPatch("no_slip_wall", 0, 0, 0);
    private BoundaryPatch bcYmax = new BoundaryPatch("no_slip_wall", 0, 0, 0);
    private BoundaryPatch bcZmin = new BoundaryPatch("no_slip_wall", 0, 0, 0);
    private BoundaryPatch bcZmax = new BoundaryPatch("moving_lid", 1.0f, 0, 0);

    public BoundaryPatch getBcXmin() { return bcXmin; }
    public void setBcXmin(BoundaryPatch bcXmin) { this.bcXmin = bcXmin; }
    public BoundaryPatch getBcXmax() { return bcXmax; }
    public void setBcXmax(BoundaryPatch bcXmax) { this.bcXmax = bcXmax; }
    public BoundaryPatch getBcYmin() { return bcYmin; }
    public void setBcYmin(BoundaryPatch bcYmin) { this.bcYmin = bcYmin; }
    public BoundaryPatch getBcYmax() { return bcYmax; }
    public void setBcYmax(BoundaryPatch bcYmax) { this.bcYmax = bcYmax; }
    public BoundaryPatch getBcZmin() { return bcZmin; }
    public void setBcZmin(BoundaryPatch bcZmin) { this.bcZmin = bcZmin; }
    public BoundaryPatch getBcZmax() { return bcZmax; }
    public void setBcZmax(BoundaryPatch bcZmax) { this.bcZmax = bcZmax; }

    public BoundaryPatch getBoundaryPatch(String face) {
        return switch (face.toLowerCase()) {
            case "xmin" -> bcXmin;
            case "xmax" -> bcXmax;
            case "ymin" -> bcYmin;
            case "ymax" -> bcYmax;
            case "zmin" -> bcZmin;
            case "zmax" -> bcZmax;
            default -> throw new IllegalArgumentException("Unknown boundary face: " + face);
        };
    }

    // Getters and Setters
    public int getNx() { return nx; }
    public void setNx(int nx) { this.nx = nx; }
    public int getNy() { return ny; }
    public void setNy(int ny) { this.ny = ny; }
    public int getNz() { return nz; }
    public void setNz(int nz) { this.nz = nz; }
    public float getDx() { return dx; }
    public void setDx(float dx) { this.dx = dx; }
    public float getDy() { return dy; }
    public void setDy(float dy) { this.dy = dy; }
    public float getDz() { return dz; }
    public void setDz(float dz) { this.dz = dz; }
    public float getDt() { return dt; }
    public void setDt(float dt) { this.dt = dt; }
    public float getNu() { return nu; }
    public void setNu(float nu) { this.nu = nu; }
    public float getRho() { return rho; }
    public void setRho(float rho) { this.rho = rho; }
    public float getUb() { return ub; }
    public void setUb(float ub) { this.ub = ub; }
    public int getNlevel() { return nlevel; }
    public void setNlevel(int nlevel) { this.nlevel = nlevel; }
    public int getMgIterations() { return mgIterations; }
    public void setMgIterations(int mgIterations) { this.mgIterations = mgIterations; }

    public boolean isEnableHeat() { return enableHeat; }
    public void setEnableHeat(boolean enableHeat) { this.enableHeat = enableHeat; }
    public boolean isEnableTurbulence() { return enableTurbulence; }
    public void setEnableTurbulence(boolean enableTurbulence) { this.enableTurbulence = enableTurbulence; }
    public boolean isEnableVof() { return enableVof; }
    public void setEnableVof(boolean enableVof) { this.enableVof = enableVof; }

    public float getThermalDiffusivity() { return thermalDiffusivity; }
    public void setThermalDiffusivity(float thermalDiffusivity) { this.thermalDiffusivity = thermalDiffusivity; }
    public float getBetaThermal() { return betaThermal; }
    public void setBetaThermal(float betaThermal) { this.betaThermal = betaThermal; }
    public float gettRef() { return tRef; }
    public void settRef(float tRef) { this.tRef = tRef; }
    public float getGx() { return gx; }
    public void setGx(float gx) { this.gx = gx; }
    public float getGy() { return gy; }
    public void setGy(float gy) { this.gy = gy; }
    public float getGz() { return gz; }
    public void setGz(float gz) { this.gz = gz; }

    public float getRho1() { return rho1; }
    public void setRho1(float rho1) { this.rho1 = rho1; }
    public float getRho2() { return rho2; }
    public void setRho2(float rho2) { this.rho2 = rho2; }
    public float getNu1() { return nu1; }
    public void setNu1(float nu1) { this.nu1 = nu1; }
    public float getNu2() { return nu2; }
    public void setNu2(float nu2) { this.nu2 = nu2; }
    public float getcAlpha() { return cAlpha; }
    public void setcAlpha(float cAlpha) { this.cAlpha = cAlpha; }

    public String getActiveField() { return activeField; }
    public void setActiveField(String activeField) { this.activeField = activeField; }
    public String getSliceAxis() { return sliceAxis; }
    public void setSliceAxis(String sliceAxis) { this.sliceAxis = sliceAxis; }
    public int getSliceIndex() { return sliceIndex; }
    public void setSliceIndex(int sliceIndex) { this.sliceIndex = sliceIndex; }
}
