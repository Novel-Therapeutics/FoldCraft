"""Measure whether JAX's persistent compilation cache eliminates the
per-trajectory design_3stage recompile (the dominant remaining cost after
Perf #1; measured to recur every trajectory, ~75-115s of compile each).

Builds a fresh design model TWICE (simulating two trajectories) with the cache
ENABLED and times design_3stage each. Trial 1 compiles (cache miss); trial 2
should LOAD from cache (hit). The gap (trial1 - trial2) is the per-trajectory
saving the cache would buy -- minus the load time, which is the unknown the
benchmark exists to measure. Results are bit-identical either way (same
executable), so this is purely a wall-clock question.

    rm -rf /tmp/jaxcache && python baseline/bench_jax_cache.py
"""
import os
import sys
import time

# Enable the on-disk compilation cache BEFORE any jax compilation happens.
import jax
CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxcache")
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)   # cache everything
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)  # incl. fast compiles
print(f"jax {jax.__version__}; compilation cache -> {CACHE_DIR}", flush=True)

sys.path.insert(0, ".")
sys.path.insert(0, "baseline")
import ab_get_best as A
from colabdesign import clear_mem

binder_template, binder_hotspots = A.FOLDS["top7"]
cond_cmap, cond_cmap_mask, binder_len = A.build_cond_cmap(binder_template, binder_hotspots)

times = []
for trial in (1, 2, 3):
    clear_mem()                      # mimic the per-trajectory buffer wipe
    t0 = time.time()
    m = A.design_model(cond_cmap, cond_cmap_mask, binder_len)
    m.design_3stage(*A.DESIGN_STAGES)
    dt = time.time() - t0
    times.append(dt)
    print(f"trial {trial}: fresh model + design_3stage = {dt:.1f}s "
          f"(final cmap_loss {float(m.aux['log']['cmap_loss_binder']):.3f})", flush=True)

print(f"\ntrial1 (compile) {times[0]:.1f}s vs trial2 (cache hit) {times[1]:.1f}s "
      f"vs trial3 {times[2]:.1f}s")
saved = times[0] - times[1]
print(f"per-trajectory saving from the cache: ~{saved:.0f}s "
      f"({100*saved/times[0]:.0f}% of design_3stage)")
print("VERDICT: cache helps" if saved > 20 else
      "VERDICT: cache gives little (load ~ compile, or jax already dedups)")
