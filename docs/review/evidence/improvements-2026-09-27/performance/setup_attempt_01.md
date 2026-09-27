# Initial profiling setup

The first attempt stopped at the telemetry import before launching inference because psutil was unavailable. No GPU model run was started. Telemetry now uses the operating system ps command; the runtime was not modified. The subsequent 12-run performance-02 benchmark completed successfully.
