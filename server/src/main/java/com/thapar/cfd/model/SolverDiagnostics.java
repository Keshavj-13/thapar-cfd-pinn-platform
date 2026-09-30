package com.thapar.cfd.model;

public class SolverDiagnostics {
    private long currentStep = 0;
    private double physicalTime = 0.0;
    private float maxDivergence = 0.0f;
    private float pressureResidual = 0.0f;
    private float massFluxError = 0.0f;
    private double lastStepMs = 0.0;
    private long vramBytes = 0;
    private boolean isRunning = false;
    private double framesPerSecond = 0.0;
    /** -1 = unbounded (Live Preview mode); >=0 = the step a bounded Run will stop at. */
    private long targetStep = -1;
    /** True once a bounded Run (see CfdEngineService#startRun) reaches its target step
     *  and auto-stops -- distinct from a user-paused mid-run state. */
    private boolean runCompleted = false;

    public long getTargetStep() { return targetStep; }
    public void setTargetStep(long targetStep) { this.targetStep = targetStep; }
    public boolean isRunCompleted() { return runCompleted; }
    public void setRunCompleted(boolean runCompleted) { this.runCompleted = runCompleted; }

    public long getCurrentStep() { return currentStep; }
    public void setCurrentStep(long currentStep) { this.currentStep = currentStep; }
    public double getPhysicalTime() { return physicalTime; }
    public void setPhysicalTime(double physicalTime) { this.physicalTime = physicalTime; }
    public float getMaxDivergence() { return maxDivergence; }
    public void setMaxDivergence(float maxDivergence) { this.maxDivergence = maxDivergence; }
    public float getPressureResidual() { return pressureResidual; }
    public void setPressureResidual(float pressureResidual) { this.pressureResidual = pressureResidual; }
    public float getMassFluxError() { return massFluxError; }
    public void setMassFluxError(float massFluxError) { this.massFluxError = massFluxError; }
    public double getLastStepMs() { return lastStepMs; }
    public void setLastStepMs(double lastStepMs) { this.lastStepMs = lastStepMs; }
    public long getVramBytes() { return vramBytes; }
    public void setVramBytes(long vramBytes) { this.vramBytes = vramBytes; }
    public boolean isRunning() { return isRunning; }
    public void setRunning(boolean running) { isRunning = running; }
    public double getFramesPerSecond() { return framesPerSecond; }
    public void setFramesPerSecond(double framesPerSecond) { this.framesPerSecond = framesPerSecond; }

    public double getVramMb() {
        return vramBytes / (1024.0 * 1024.0);
    }
}
