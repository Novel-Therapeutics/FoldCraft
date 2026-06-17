"""Memory-gated, multi-GPU scheduler for FoldCraft design campaigns.

FoldCraft design jobs are embarrassingly parallel: each fold's trajectories are
independent, and trajectories of different folds are independent. This scheduler
exploits that to pack several designs onto one GPU when memory permits, and to
fan a campaign out across many GPUs (e.g. a cloud node of 8x A100) -- the same
machinery serves both.

Design
------
* Unit of work is a **chunk**: ``FoldCraft.py --num_designs <chunk_traj>`` for one
  fold, written to its own dir ``<repro>/<fold>__c<idx>``. FoldCraft.py has no
  trajectory-offset arg and names trajectories ``traj_1..N``, so two chunks of the
  same fold would collide -- separate dirs avoid that, and merge_chunks() stitches
  them back together with chunk-tagged names.
* **Memory gating**: a chunk launches on a GPU only when that GPU's *estimated*
  free memory (capacity minus already-committed jobs) AND its *actual* free memory
  (from nvidia-smi) both clear ``job_mem + headroom``. The double check guards
  against both over-commit and a newly-launched neighbour still ramping up.
* **Resume**: FoldCraft.py writes ``results.csv`` only after a chunk finishes all
  its trajectories, so ``results.csv`` existing == chunk done. Completed chunks
  (and already-merged folds) are skipped, making the whole campaign restartable
  without losing finished work.
* **Loud failure**: a chunk whose process exits non-zero is recorded as failed and
  (once) retried solo with full headroom; a fold is merged only when all its
  chunks succeeded. Nothing is silently dropped.

The functions above the ``# --- orchestration ---`` line are pure and GPU-free so
they can be unit-tested (see tests/test_scheduler.py).
"""
import os
import shutil
import subprocess
import sys
import time

import pandas as pd


# ---------------------------------------------------------------------------
# Pure planning logic (no GPU, no subprocess) -- unit-tested
# ---------------------------------------------------------------------------
class Chunk:
    """One FoldCraft.py invocation: ``chunk_traj`` trajectories of ``fold``."""

    def __init__(self, fold, idx, chunk_traj, mem_gb, spec):
        self.fold = fold
        self.idx = idx
        self.chunk_traj = chunk_traj
        self.mem_gb = mem_gb
        self.spec = spec  # dict: template, target, target_hotspots, binder_hotspots
        self.tag = f"{fold}__c{idx}"

    def out_dir(self, repro):
        return os.path.join(repro, self.tag)

    def __repr__(self):
        return f"Chunk({self.tag}, traj={self.chunk_traj}, mem={self.mem_gb}GB)"


def plan_chunks(folds, chunk_traj):
    """Split each fold's trajectories into chunks of at most ``chunk_traj``.

    ``folds`` is a list of dicts with keys: fold, template, target,
    target_hotspots, binder_hotspots, total_traj, mem_gb. Returns a flat list of
    Chunk objects, largest-memory-first so the tight-fitting jobs schedule before
    the GPU fills with small ones (a simple but effective bin-packing heuristic).
    """
    if chunk_traj < 1:
        raise ValueError(f"chunk_traj must be >= 1, got {chunk_traj}")
    chunks = []
    for f in folds:
        total = int(f["total_traj"])
        if total < 1:
            raise ValueError(f"{f['fold']}: total_traj must be >= 1, got {total}")
        idx = 0
        remaining = total
        while remaining > 0:
            n = min(chunk_traj, remaining)
            chunks.append(Chunk(f["fold"], idx, n, float(f["mem_gb"]),
                                {k: f[k] for k in ("template", "target",
                                                   "target_hotspots", "binder_hotspots")}))
            remaining -= n
            idx += 1
    chunks.sort(key=lambda c: c.mem_gb, reverse=True)
    return chunks


def chunk_done(chunk, repro):
    """A chunk is finished iff its output dir has a results.csv (written only at
    the end of FoldCraft.py's loop)."""
    return os.path.exists(os.path.join(chunk.out_dir(repro), "results.csv"))


def fold_done(fold, repro):
    """A fold is finished iff its merged dir has a results.csv."""
    return os.path.exists(os.path.join(repro, fold, "results.csv"))


def filter_todo(chunks, repro):
    """Drop chunks whose fold is already merged or whose own chunk is complete."""
    return [c for c in chunks
            if not fold_done(c.fold, repro) and not chunk_done(c, repro)]


def fits(actual_free_gb, committed_gb, capacity_gb, job_mem_gb, headroom_gb):
    """True if a job of ``job_mem_gb`` fits on a GPU given both the committed
    budget (capacity minus running jobs) and the measured free memory."""
    est_free = capacity_gb - committed_gb
    return (est_free >= job_mem_gb + headroom_gb
            and actual_free_gb >= job_mem_gb + headroom_gb)


def remap_name(name, chunk_idx):
    """Chunk-tag a trajectory/design name so merged chunks don't collide."""
    return f"c{chunk_idx}_{name}"


def merge_chunks(fold, chunk_dirs, out_dir):
    """Stitch a fold's chunk dirs into one scoreable dir with unique names.

    Concatenates each chunk's results.csv (re-tagging the ``name`` column) and
    copies the matching designs/<name>.pdb (+ .pickle, needed for ipSAE) under the
    re-tagged name. Returns the merged row count. Raises if a chunk lacks
    results.csv or a referenced design file is missing (fail loud).
    """
    designs_out = os.path.join(out_dir, "designs")
    os.makedirs(designs_out, exist_ok=True)
    frames = []
    for idx, cdir in enumerate(chunk_dirs):
        csv = os.path.join(cdir, "results.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(f"merge {fold}: chunk missing results.csv: {csv}")
        df = pd.read_csv(csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        for old in df["name"]:
            new = remap_name(old, idx)
            for ext in (".pdb", ".pickle"):
                src = os.path.join(cdir, "designs", f"{old}{ext}")
                if ext == ".pdb" and not os.path.exists(src):
                    raise FileNotFoundError(
                        f"merge {fold}: design PDB missing: {src}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(designs_out, f"{new}{ext}"))
        df["name"] = df["name"].map(lambda n: remap_name(n, idx))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    return len(merged)


# ---------------------------------------------------------------------------
# orchestration (GPU + subprocess) -- not unit-tested
# ---------------------------------------------------------------------------
def _nvidia_smi():
    for cand in ("nvidia-smi", "/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        if shutil.which(cand) or os.path.exists(cand):
            return cand
    raise RuntimeError("nvidia-smi not found")


def discover_gpus(smi):
    out = subprocess.check_output([smi, "-L"], text=True)
    return [ln.split(":")[0].split()[-1] for ln in out.strip().splitlines() if ln.strip()]


def gpu_mem_gb(smi, gpu):
    """(free_gb, total_gb) for one GPU index."""
    out = subprocess.check_output(
        [smi, "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits", "-i", str(gpu)], text=True)
    free_mib, total_mib = (int(x) for x in out.strip().split(","))
    return free_mib / 1024.0, total_mib / 1024.0


def launch(chunk, repro, gpu, repo, python):
    out = chunk.out_dir(repro)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONUNBUFFERED="1")
    log = open(os.path.join(out, "chunk.log"), "w")
    cmd = [python, "FoldCraft.py",
           "--output_folder", out,
           "--binder_template", chunk.spec["template"],
           "--target_template", chunk.spec["target"],
           "--target_hotspots", chunk.spec["target_hotspots"],
           "--binder_hotspots", chunk.spec["binder_hotspots"],
           "--num_designs", str(chunk.chunk_traj)]
    return subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT), log


def run(folds, repro, repo, gpus, chunk_traj, headroom_gb, python,
        poll_s=15, log=print):
    """Schedule all folds' chunks across ``gpus`` with memory gating, then merge."""
    smi = _nvidia_smi()
    caps = {g: gpu_mem_gb(smi, g)[1] for g in gpus}
    committed = {g: 0.0 for g in gpus}          # GB reserved by running jobs
    running = {}                                  # pid -> (proc, log, chunk, gpu)

    queue = filter_todo(plan_chunks(folds, chunk_traj), repro)
    total = len(plan_chunks(folds, chunk_traj))
    log(f"[sched] {len(queue)}/{total} chunks to run on GPUs {gpus} "
        f"(rest already complete)")
    failed, retried = [], set()

    while queue or running:
        # try to launch as many queued jobs as fit
        progress = True
        while progress and queue:
            progress = False
            for gi, g in enumerate(gpus):
                if not queue:
                    break
                # pick the largest queued job that fits this GPU
                free, _ = gpu_mem_gb(smi, g)
                for j, c in enumerate(queue):
                    if fits(free, committed[g], caps[g], c.mem_gb, headroom_gb):
                        proc, lf = launch(c, repro, g, repo, python)
                        committed[g] += c.mem_gb
                        running[proc.pid] = (proc, lf, c, g)
                        queue.pop(j)
                        log(f"[sched] launch {c.tag} -> gpu{g} "
                            f"(free {free:.1f}GB, committed {committed[g]:.1f}GB)")
                        progress = True
                        break
        # reap finished jobs
        time.sleep(poll_s)
        for pid in list(running):
            proc, lf, c, g = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            committed[g] -= c.mem_gb
            del running[pid]
            if rc == 0 and chunk_done(c, repro):
                log(f"[sched] DONE {c.tag} (gpu{g}, rc=0)")
            elif c.tag not in retried:
                # one solo retry with full headroom (memory pressure is the
                # most common cause of a failed pack)
                log(f"[sched] FAIL {c.tag} (rc={rc}); will retry solo")
                retried.add(c.tag)
                queue.append(c)
            else:
                log(f"[sched] FAIL {c.tag} (rc={rc}) on retry -- giving up")
                failed.append(c)

    # merge folds whose chunks all completed
    merged = []
    for f in folds:
        if fold_done(f["fold"], repro):
            continue
        cdirs = sorted(
            d.out_dir(repro) for d in plan_chunks([f], chunk_traj))
        if all(os.path.exists(os.path.join(d, "results.csv")) for d in cdirs):
            n = merge_chunks(f["fold"], cdirs, os.path.join(repro, f["fold"]))
            merged.append((f["fold"], n))
            log(f"[sched] merged {f['fold']}: {n} designs")
        else:
            log(f"[sched] NOT merging {f['fold']}: some chunks failed")

    if failed:
        raise SystemExit(f"[sched] {len(failed)} chunk(s) failed: "
                         f"{[c.tag for c in failed]}")
    return merged


# ---------------------------------------------------------------------------
# config + CLI
# ---------------------------------------------------------------------------
def load_config(path):
    """TSV: fold, template, target, target_hotspots, binder_hotspots, total_traj, mem_gb."""
    folds = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            folds.append(dict(zip(header, line.split("\t"))))
    return folds


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="TSV config (see load_config)")
    p.add_argument("--repro", default="baseline/repro", help="output root")
    p.add_argument("--repo", default=os.getcwd(), help="FoldCraft repo root")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpus", default="auto",
                   help="comma list of GPU indices, or 'auto'")
    p.add_argument("--chunk-traj", type=int, default=10,
                   help="trajectories per chunk (smaller = finer load balance)")
    p.add_argument("--headroom-gb", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the schedule plan and exit (no GPU needed)")
    args = p.parse_args(argv)

    folds = load_config(args.config)
    if args.dry_run:
        chunks = plan_chunks(folds, args.chunk_traj)
        todo = filter_todo(chunks, args.repro)
        print(f"folds={len(folds)} chunks={len(chunks)} todo={len(todo)} "
              f"(done={len(chunks) - len(todo)})")
        for c in todo:
            print(f"  {c!r}")
        return

    smi = _nvidia_smi()
    gpus = discover_gpus(smi) if args.gpus == "auto" else args.gpus.split(",")
    run(folds, args.repro, args.repo, gpus, args.chunk_traj,
        args.headroom_gb, args.python)


if __name__ == "__main__":
    main()
