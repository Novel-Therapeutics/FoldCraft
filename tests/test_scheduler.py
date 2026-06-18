"""Tests for the pure planning logic of baseline/scheduler.py.

These cover the parts that decide *what runs where* and *how chunks are stitched
back together* -- the logic whose bugs would silently corrupt a campaign
(double-running work, dropping designs, or colliding names on merge). The GPU
orchestration (subprocess/nvidia-smi) is not exercised here.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
import scheduler as sch


def _fold(name, total, mem=12):
    return {"fold": name, "template": f"{name}.pdb", "target": "t.pdb",
            "target_hotspots": "1-5", "binder_hotspots": "1-5",
            "total_traj": total, "mem_gb": mem}


# --- plan_chunks -----------------------------------------------------------
class TestPlanChunks:
    def test_splits_into_ceil_chunks(self):
        chunks = sch.plan_chunks([_fold("a", 40)], chunk_traj=10)
        assert len(chunks) == 4
        assert [c.chunk_traj for c in chunks] == [10, 10, 10, 10]

    def test_remainder_chunk_is_smaller(self):
        chunks = sch.plan_chunks([_fold("a", 25)], chunk_traj=10)
        assert sorted(c.chunk_traj for c in chunks) == [5, 10, 10]
        # total trajectories are preserved
        assert sum(c.chunk_traj for c in chunks) == 25

    def test_largest_memory_first(self):
        chunks = sch.plan_chunks([_fold("small", 10, mem=8),
                                  _fold("big", 10, mem=18)], chunk_traj=10)
        assert chunks[0].fold == "big" and chunks[-1].fold == "small"

    def test_chunk_tags_unique_within_fold(self):
        chunks = sch.plan_chunks([_fold("a", 30)], chunk_traj=10)
        assert len({c.tag for c in chunks}) == 3
        assert {c.tag for c in chunks} == {"a__c0", "a__c1", "a__c2"}

    def test_rejects_bad_chunk_size(self):
        with pytest.raises(ValueError):
            sch.plan_chunks([_fold("a", 10)], chunk_traj=0)

    def test_rejects_bad_total(self):
        with pytest.raises(ValueError):
            sch.plan_chunks([_fold("a", 0)], chunk_traj=10)


# --- fits (memory gating) --------------------------------------------------
class TestFits:
    def test_fits_when_both_budgets_clear(self):
        # 24GB card, nothing committed, 11GB job + 2GB headroom -> fits
        assert sch.fits(actual_free_gb=22, committed_gb=0, capacity_gb=24,
                        job_mem_gb=11, headroom_gb=2)

    def test_blocked_by_committed_budget_even_if_smi_reports_free(self):
        # a neighbour just launched but hasn't ramped its memory yet: nvidia-smi
        # still shows lots free, but the committed budget must block over-pack.
        assert not sch.fits(actual_free_gb=22, committed_gb=20, capacity_gb=24,
                            job_mem_gb=11, headroom_gb=2)

    def test_blocked_by_actual_free_even_if_budget_says_ok(self):
        # something outside our accounting is using the card -> respect reality.
        assert not sch.fits(actual_free_gb=5, committed_gb=0, capacity_gb=24,
                            job_mem_gb=11, headroom_gb=2)

    def test_two_small_jobs_pack_on_one_card(self):
        cap, head = 24, 2
        # first 11GB job fits (nothing committed yet, card nearly empty)
        assert sch.fits(22, 0, cap, 11, head)
        # after job1 ramped to 11GB, actual free is 13GB and committed is 11GB;
        # a second 11GB job needs 11+2=13 of both budgets -> fits exactly,
        # leaving the 2GB headroom as global slack.
        assert sch.fits(13, 11, cap, 11, head)
        # a third job cannot: only 2GB of committed budget / actual free remain.
        assert not sch.fits(2, 22, cap, 11, head)


# --- unschedulable preflight (P2: no infinite spin) ------------------------
class TestUnschedulable:
    def test_flags_chunk_bigger_than_largest_gpu(self):
        # tim needs 17+2=19GB; a 16GB card can never run it.
        chunks = sch.plan_chunks([_fold("tim", 10, mem=17)], chunk_traj=10)
        bad = sch.unschedulable_chunks(chunks, max_capacity_gb=16, headroom_gb=2)
        assert [c.fold for c in bad] == ["tim"]

    def test_empty_when_everything_fits(self):
        chunks = sch.plan_chunks([_fold("tim", 10, mem=17)], chunk_traj=10)
        assert sch.unschedulable_chunks(chunks, max_capacity_gb=24, headroom_gb=2) == []

    def test_boundary_exact_fit_is_schedulable(self):
        # 22+2 == 24 exactly -> schedulable (not flagged)
        chunks = sch.plan_chunks([_fold("x", 10, mem=22)], chunk_traj=10)
        assert sch.unschedulable_chunks(chunks, max_capacity_gb=24, headroom_gb=2) == []


# --- can_launch (P2: true solo retry) --------------------------------------
class TestCanLaunch:
    def _chunk(self, mem=11, solo=False):
        c = sch.plan_chunks([_fold("a", 10, mem=mem)], chunk_traj=10)[0]
        c.solo = solo
        return c

    def test_normal_chunk_packs_alongside_others(self):
        c = self._chunk(mem=11)
        # 13GB free, 11GB already committed by a neighbour -> still packs
        assert sch.can_launch(c, 13, 11, 24, 2)

    def test_solo_chunk_refused_when_gpu_busy(self):
        c = self._chunk(mem=11, solo=True)
        # even though 11GB would fit memory-wise, a solo retry must wait for an
        # empty GPU (committed > 0 blocks it)
        assert not sch.can_launch(c, 13, 11, 24, 2)

    def test_solo_chunk_runs_on_empty_gpu(self):
        c = self._chunk(mem=11, solo=True)
        assert sch.can_launch(c, 24, 0, 24, 2)


# --- resume detection ------------------------------------------------------
def _merged(repro, fold, with_template=True):
    """Create a merged fold dir (results.csv, optionally template.pdb)."""
    d = os.path.join(repro, fold)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "results.csv"), "w").close()
    if with_template:
        open(os.path.join(d, "template.pdb"), "w").close()


class TestResume:
    def test_skips_completed_chunk_and_scoreable_fold(self, tmp_path):
        repro = str(tmp_path)
        folds = [_fold("a", 20), _fold("b", 20)]
        chunks = sch.plan_chunks(folds, chunk_traj=10)
        # a__c0 chunk done (results.csv); fold b fully merged AND scoreable
        os.makedirs(os.path.join(repro, "a__c0"))
        open(os.path.join(repro, "a__c0", "results.csv"), "w").close()
        _merged(repro, "b", with_template=True)

        todo = sch.filter_todo(chunks, repro)
        assert {c.tag for c in todo} == {"a__c1"}  # a__c0 + both b chunks skipped

    def test_fold_with_results_but_no_template_is_not_done(self, tmp_path):
        # a fold merged before merge_chunks recorded the template: results.csv but
        # no template.pdb -> NOT scoreable, must not be treated as done/skipped.
        repro = str(tmp_path)
        _merged(repro, "b", with_template=False)
        assert not sch.fold_done("b", repro)
        # with the template it is done
        open(os.path.join(repro, "b", "template.pdb"), "w").close()
        assert sch.fold_done("b", repro)


# --- repair_fold_template (P2: keep merged folds scoreable) ----------------
class TestRepairFoldTemplate:
    def test_repairs_missing_template(self, tmp_path):
        repro = str(tmp_path)
        _merged(repro, "b", with_template=False)
        src = os.path.join(repro, "src_template.pdb")
        with open(src, "w") as fh:
            fh.write("TEMPLATE")
        assert sch.repair_fold_template(os.path.join(repro, "b"), src) is True
        copied = os.path.join(repro, "b", "template.pdb")
        assert os.path.exists(copied) and open(copied).read() == "TEMPLATE"

    def test_noop_when_template_present(self, tmp_path):
        repro = str(tmp_path)
        _merged(repro, "b", with_template=True)
        src = os.path.join(repro, "src.pdb"); open(src, "w").close()
        assert sch.repair_fold_template(os.path.join(repro, "b"), src) is False

    def test_noop_when_not_merged(self, tmp_path):
        # no results.csv -> nothing to repair (the fold isn't merged yet)
        repro = str(tmp_path)
        os.makedirs(os.path.join(repro, "b"))
        src = os.path.join(repro, "src.pdb"); open(src, "w").close()
        assert sch.repair_fold_template(os.path.join(repro, "b"), src) is False

    def test_raises_when_source_template_missing(self, tmp_path):
        repro = str(tmp_path)
        _merged(repro, "b", with_template=False)
        with pytest.raises(FileNotFoundError):
            sch.repair_fold_template(os.path.join(repro, "b"),
                                     os.path.join(repro, "nope.pdb"))


# --- repair ordering (P2: repair before filter_todo, not after scheduling) -
class TestRepairBeforeFilter:
    def test_repair_marks_fold_done_so_chunks_are_not_requeued(self, tmp_path):
        # Reproduces the bug: a fold merged by an older scheduler (results.csv,
        # NO template.pdb) with its chunk dirs gone. Without an up-front repair,
        # filter_todo would re-queue (and the run would re-run) its chunks.
        repro = str(tmp_path / "repro"); os.makedirs(repro)
        src = str(tmp_path / "tmpl.pdb"); open(src, "w").write("X")
        folds = [_fold("top7", 20)]
        folds[0]["template"] = src
        _merged(repro, "top7", with_template=False)  # results.csv only
        chunks = sch.plan_chunks(folds, chunk_traj=10)

        # before repair: both chunks are wrongly considered outstanding work
        assert {c.tag for c in sch.filter_todo(chunks, repro)} == {"top7__c0", "top7__c1"}

        # repairing first restores the template -> fold is done -> chunks skipped
        assert sch.repair_merged_folds(folds, repro) == ["top7"]
        assert os.path.exists(os.path.join(repro, "top7", "template.pdb"))
        assert sch.filter_todo(chunks, repro) == []

    def test_no_repair_when_template_already_present(self, tmp_path):
        repro = str(tmp_path / "repro"); os.makedirs(repro)
        src = str(tmp_path / "tmpl.pdb"); open(src, "w").close()
        folds = [_fold("top7", 20)]; folds[0]["template"] = src
        _merged(repro, "top7", with_template=True)
        assert sch.repair_merged_folds(folds, repro) == []


# --- resolve_config_path (P3: CONFIG resolves against --repo) ---------------
class TestResolveConfigPath:
    def test_relative_config_resolved_against_repo(self):
        assert sch.resolve_config_path("baseline/c.tsv", "/abs/repo") == \
            "/abs/repo/baseline/c.tsv"

    def test_absolute_config_passes_through(self):
        assert sch.resolve_config_path("/data/c.tsv", "/abs/repo") == "/data/c.tsv"

    def test_main_dry_run_with_relative_config_from_foreign_cwd(
            self, tmp_path, monkeypatch, capsys):
        # The reviewer's exact scenario: invoke from OUTSIDE the repo with a
        # repo-relative config + --repo. Must load the config, not FileNotFoundError.
        repo = tmp_path / "repo"; (repo / "baseline").mkdir(parents=True)
        (repo / "baseline" / "c.tsv").write_text(
            "fold\ttemplate\ttarget\ttarget_hotspots\tbinder_hotspots\ttotal_traj\tmem_gb\n"
            "top7\texamples/t.pdb\texamples/g.pdb\t1\t1\t40\t11\n")
        monkeypatch.chdir(tmp_path)  # a cwd where "baseline/c.tsv" does NOT exist
        sch.main(["baseline/c.tsv", "--repo", str(repo), "--dry-run"])
        assert "folds=1" in capsys.readouterr().out


# --- resolve_paths (P1: scheduler/child cwd agreement) ---------------------
class TestResolvePaths:
    def test_relative_paths_resolved_against_repo_not_cwd(self):
        folds = [{"fold": "a", "template": "examples/templates/1qys1.pdb",
                  "target": "examples/targets/pd-l1-1.pdb",
                  "target_hotspots": "1", "binder_hotspots": "1",
                  "total_traj": 40, "mem_gb": 11}]
        out, repo, repro = sch.resolve_paths(folds, "/abs/repo", "baseline/repro")
        assert repo == "/abs/repo"
        # repro + template + target resolve against repo, not the scheduler cwd
        assert repro == "/abs/repo/baseline/repro"
        assert out[0]["template"] == "/abs/repo/examples/templates/1qys1.pdb"
        assert out[0]["target"] == "/abs/repo/examples/targets/pd-l1-1.pdb"

    def test_absolute_paths_pass_through(self):
        folds = [{"fold": "a", "template": "/data/t.pdb", "target": "/data/g.pdb",
                  "target_hotspots": "1", "binder_hotspots": "1",
                  "total_traj": 40, "mem_gb": 11}]
        out, repo, repro = sch.resolve_paths(folds, "/abs/repo", "/scratch/out")
        assert repro == "/scratch/out"
        assert out[0]["template"] == "/data/t.pdb"
        assert out[0]["target"] == "/data/g.pdb"

    def test_idempotent(self):
        folds = [{"fold": "a", "template": "/data/t.pdb", "target": "/data/g.pdb",
                  "target_hotspots": "1", "binder_hotspots": "1",
                  "total_traj": 40, "mem_gb": 11}]
        once = sch.resolve_paths(folds, "/abs/repo", "/scratch/out")
        twice = sch.resolve_paths(once[0], once[1], once[2])
        assert twice[1:] == once[1:] and twice[0] == once[0]


# --- merge_chunks ----------------------------------------------------------
class TestMergeChunks:
    def _make_chunk(self, root, tag, names):
        cdir = os.path.join(root, tag)
        os.makedirs(os.path.join(cdir, "designs"))
        pd.DataFrame({"name": names, "plddt": [0.8] * len(names),
                      "ipae": [0.3] * len(names)}).to_csv(
            os.path.join(cdir, "results.csv"), index=False)
        for n in names:
            for ext in (".pdb", ".pickle"):
                open(os.path.join(cdir, "designs", f"{n}{ext}"), "w").close()
        return cdir

    def _template(self, root):
        path = os.path.join(root, "tmpl.pdb")
        open(path, "w").close()
        return path

    def test_merge_concats_and_retags_without_collision(self, tmp_path):
        root = str(tmp_path)
        # both chunks independently produced traj_1_0 / traj_1_1 -- would collide
        c0 = self._make_chunk(root, "a__c0", ["traj_1_0", "traj_1_1"])
        c1 = self._make_chunk(root, "a__c1", ["traj_1_0", "traj_1_1"])
        out = os.path.join(root, "a")
        n = sch.merge_chunks("a", [c0, c1], out, self._template(root))

        assert n == 4
        df = pd.read_csv(os.path.join(out, "results.csv"))
        # names are unique after chunk-tagging
        assert df["name"].nunique() == 4
        assert set(df["name"]) == {"c0_traj_1_0", "c0_traj_1_1",
                                   "c1_traj_1_0", "c1_traj_1_1"}
        # every results row has a matching design PDB + pickle under the new name
        for nm in df["name"]:
            assert os.path.exists(os.path.join(out, "designs", f"{nm}.pdb"))
            assert os.path.exists(os.path.join(out, "designs", f"{nm}.pickle"))
        # the merged fold is self-describing: template recorded for RMSD scoring
        assert os.path.exists(os.path.join(out, "template.pdb"))

    def test_merge_fails_loud_on_missing_design(self, tmp_path):
        root = str(tmp_path)
        cdir = os.path.join(root, "a__c0")
        os.makedirs(os.path.join(cdir, "designs"))
        pd.DataFrame({"name": ["traj_1_0"], "plddt": [0.8]}).to_csv(
            os.path.join(cdir, "results.csv"), index=False)
        # results.csv references traj_1_0 but no PDB exists -> must raise
        with pytest.raises(FileNotFoundError):
            sch.merge_chunks("a", [cdir], os.path.join(root, "a"), self._template(root))

    def test_merge_fails_loud_on_missing_results(self, tmp_path):
        root = str(tmp_path)
        cdir = os.path.join(root, "a__c0")
        os.makedirs(cdir)
        with pytest.raises(FileNotFoundError):
            sch.merge_chunks("a", [cdir], os.path.join(root, "a"), self._template(root))

    def test_merge_fails_loud_on_missing_template(self, tmp_path):
        root = str(tmp_path)
        c0 = self._make_chunk(root, "a__c0", ["traj_1_0"])
        # template path does not exist -> must raise (merged fold would be
        # unscoreable for RMSD)
        with pytest.raises(FileNotFoundError):
            sch.merge_chunks("a", [c0], os.path.join(root, "a"),
                             os.path.join(root, "nope.pdb"))
