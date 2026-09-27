"""Render the recorded benchmark results without changing promotion rules."""
import argparse,json
from pathlib import Path


def write(performance,expanded,pipeline,destination):
    perf=json.loads((performance/'promotion.json').read_text())
    raw=json.loads((performance/'report.json').read_text())
    exp=json.loads((expanded/'analysis.json').read_text())
    execution=json.loads(pipeline.read_text())
    if execution['status']!='passed' or perf['status']!='passed' or exp['status']!='complete':raise ValueError('Cannot report incomplete results')
    lines=['# Performance results — 27 September 2026','',
           'Baseline: `validated-baseline-2026-09-27` (`c5e3f78`). Protocol/code: `3d293b5`.',
           '', 'Compact storage '+('passed' if perf['promote_compact'] else 'did not pass')+' the prespecified promotion gate. Scientific predictions are unchanged.', '',
           '| Complex | Full candidate MB | Compact candidate MB | Reduction | Full warm seconds | Compact warm seconds |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in perf['comparisons']:
        lines.append(f'| {r["case"]} | {r["full_candidate_bytes"]/1e6:.2f} | {r["compact_candidate_bytes"]/1e6:.2f} | {100*r["storage_reduction"]:.2f}% | {r["full_warm_seconds"]:.2f} | {r["compact_warm_seconds"]:.2f} |')
    lines += ['', 'Bytes cover four candidate predictions (two candidates × two models). Trajectory artifacts are also compacted. MB are decimal.', '',
              '| Complex | Mode | Cold seconds | Warm host RSS GiB | Warm device used GiB |', '| --- | --- | ---: | ---: | ---: |']
    for case in ('short','medium','long'):
        for mode in ('full','compact'):
            cold=next(x for x in raw['runs'] if x['case']==case and x['mode']==mode and x['phase']=='cold')
            warm=next(x for x in raw['runs'] if x['case']==case and x['mode']==mode and x['phase']=='warm')
            lines.append(f'| {case} | {mode} | {cold["seconds"]:.2f} | {warm["peak_process_tree_rss_bytes"]/2**30:.2f} | {warm["peak_device_used_bytes"]/2**30:.2f} |')
    lines += ['', 'Every other mode/phase was compared with its cold full-artifact reference: sequences, decisions, metrics, PAE and unrounded coordinates agree exactly. PAE/coordinate maximum deltas are zero. All 249 CPU tests passed on macOS and the validated Linux runtime. They exercise downstream array access and model-specific receipts. A final GPU run without an explicit storage flag used compact output and passed exact replay against full artifacts.', '',
              'The demonstrated gain is storage reduction. These single measurements do not establish a speedup or a reliable memory reduction. Host RSS is sampled process-tree RSS and can double-count shared pages; device memory includes its baseline. See the JSON for sample counts.', '',
              'Full diagnostic tensors remain available with `--artifact_mode full`; compact arrays are not quantized. No model reuse or deduplication speedup is claimed.', '',
              'Evidence: [improvements-2026-09-27](docs/review/evidence/improvements-2026-09-27/).']
    (destination/'PERFORMANCE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    lines=['# Expanded structural benchmark results — 27 September 2026','',
           '**Scientific defaults remain unchanged.** The holdout evidence does not satisfy the predeclared promotion criteria.', '',
           f'Completed {exp["paired_blocks"]} paired case/seed blocks, {exp["candidates"]} candidates, 48 full design trajectories, 192 AF2 model predictions, 96 ESMFold evaluations and 192 Boltz predictions (two repeats per candidate).', '',
           'Four targets, three scaffolds and two seeds expand the original pilot. PD-L1/PD-1 are grouped as one immune-checkpoint family. IFNAR was held out from outcome-driven method selection; no ranking threshold or temperature was tuned against it. Only one unseen family is available. There are no assay labels, so the endpoints measure structural consistency, not binding accuracy. Evaluation uses the bundled PDB constructs; full-length target context was not separately tested.', '',
           '| Split | Arm | Blocks | Independent proxy pass | Template RMSD Å | ESMFold RMSD Å | Boltz ipTM | Epitope coverage | Clashes |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in exp['summaries']:
        m=r['metrics'];lines.append(f'| {r["split"]} | {r["arm"]} | {r["paired_blocks"]} | {100*m["proxy_pass"]:.1f}% | {m["rmsd"]:.3f} | {m["esmfold_rmsd"]:.3f} | {m["boltz2_iptm"]:.3f} | {100*m["epitope_coverage"]:.1f}% | {m["interchain_clashes"]:.2f} |')
    lines += ['', 'The independent proxy requires ESMFold confidence ≥70, ESMFold/design RMSD ≤3.5 Å and mean repeated Boltz ipTM ≥0.6. It is not a biological binding label. Siblings are averaged within each trajectory before arm means.', '',
              '## Held-out paired changes (temperature 0.2 minus 0.1)', '', '| Metric | Mean change | Conditional 95% interval |', '| --- | ---: | --- |']
    for key,v in exp['temperature_deltas']['holdout'].items():
        lo,hi=v['conditional_95_interval'];lines.append(f'| {key} | {v["mean"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
    lines += ['', 'These intervals resample case-level paired means after averaging the two seeds. They condition on the observed scaffold/target panel; one unseen family cannot support population-level generalization. See `family_means` for grouped outcomes.', '',
              '## Fixed-quota ranking', '', '| Split | Policy | Selected | Independent proxy pass | Template RMSD Å |', '| --- | --- | ---: | ---: | ---: |']
    for r in exp['ranking']:
        lines.append(f'| {r["split"]} | {r["policy"]} | {r["selected"]} | {100*r["metrics"]["proxy_pass"]:.1f}% | {r["metrics"]["rmsd"]:.3f} |')
    lines += ['', 'Both policies select one candidate per case/seed from the same four-candidate pool. The baseline ranks by AF2 ipTM. The candidate rule first minimizes violations of template RMSD, clashes and epitope coverage, then ranks by ipTM. Neither consumes ESMFold/Boltz outcomes. Coverage of ordinary single-/two-model confidence gates appears below and in JSON.', '',
              '## Promotion decision', '']
    for key in ('temperature_promotion','ranking_promotion'):
        result=exp[key];lines.append(f'- {key}: '+('promote' if result['promote'] else 'retain existing default')+'. Failed gates: '+', '.join(k for k,v in result['gates'].items() if not v)+'.')
    temperature={r['arm']:r['metrics'] for r in exp['summaries'] if r['split']=='holdout'}
    ranking={r['policy']:r['metrics'] for r in exp['ranking'] if r['split']=='holdout'}
    a,b=temperature['baseline'],temperature['mpnn_temp_02']
    x,y=ranking['af2_rank'],ranking['fold_contact_rank']
    lines += ['', '## Interpretation', '',
              f'Temperature 0.2 left the held-out primary proxy unchanged ({100*a["proxy_pass"]:.1f}% versus {100*b["proxy_pass"]:.1f}%). Template and ESMFold RMSD means each improved by about {a["rmsd"]-b["rmsd"]:.2f} Å, but their conditional intervals include no improvement. Mean Boltz ipTM and epitope coverage did not improve. Both arms produced 48 unique binder sequences in this panel, so the higher temperature did not increase the observed unique-sequence count.', '',
              f'The ranking rule also left held-out proxy yield unchanged ({100*x["proxy_pass"]:.1f}% versus {100*y["proxy_pass"]:.1f}%). It reduced mean clashes from {x["interchain_clashes"]:.2f} to {y["interchain_clashes"]:.2f}, while ESMFold/design RMSD worsened from {x["esmfold_rmsd"]:.2f} to {y["esmfold_rmsd"]:.2f} Å. Better values for the features used to rank candidates did not translate into better independent proxy yield.', '',
              'Zero-width primary-proxy intervals reflect zero changes in the observed case-level means; they do not establish zero uncertainty for new targets or binding assays.', '',
              '## Confidence-gate coverage (descriptive)', '',
              '| Split | Gate | Selected / pool | Proxy-positive fraction among selected |',
              '| --- | --- | ---: | ---: |']
    for r in exp['confidence_selection']:
        fraction='unmeasured' if r['proxy_pass_fraction'] is None else f'{100*r["proxy_pass_fraction"]:.1f}%'
        lines.append(f'| {r["split"]} | {r["policy"]} | {r["selected"]} / {r["total"]} | {fraction} |')
    lines += ['', 'On the held-out panel the two-model gate narrowed selection from eight candidates to one, and the surviving candidate did not pass the independent proxy. This supports keeping the two-model acceptance policy optional rather than promoting it from its development-set behavior. These small descriptive counts do not establish binding precision or population-level recall.']
    lines += ['', 'No post-hoc threshold change was used to rescue either method. Before independent scoring, the analyzer was corrected to enforce the fold/clash ranking limits already specified in the frozen plan. The original source, both hashes and the timing are retained in `analysis_implementation_correction.json`. The normalized-loss arm rejected by the earlier pilot was not rerun.', '', '## Cost', '', '| Stage | Allocated GPU/runner wall minutes |', '| --- | ---: |']
    for stage in execution['stages']:lines.append(f'| {stage["stage"]} | {stage["seconds"]/60:.2f} |')
    lines += ['', 'Generation used up to two concurrent owned jobs; Boltz used two independent scorer processes. These are campaign wall times, not additive per-candidate GPU costs. Equal-draw quality is measured; equal-GPU-time search efficiency is not claimed.', '',
              'All candidates, including rejects, were independently scored. Paired Boltz seeds exclude the arm identity. Exact paired backbone hashes, per-model artifact checks, scorer-signature checks and raw Boltz selection hashes passed independent audits. Evaluator package/source fingerprints match before and after the campaign.', '',
              'Evidence: [improvements-2026-09-27](docs/review/evidence/improvements-2026-09-27/). AF2/Boltz structures and full optimization histories remain on Bizon. ESMFold retains scores and checkpoint/input provenance, but not monomer coordinate files; reproducing its RMSDs requires rerunning the recorded evaluator. More untouched families and assay-defined labels are needed before a scientific default or binding-accuracy claim can be promoted.']
    (destination/'EXPANDED_BENCHMARK_RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--performance',type=Path,required=True);p.add_argument('--expanded',type=Path,required=True);p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();write(a.performance,a.expanded,a.pipeline,a.destination)
