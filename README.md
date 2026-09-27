<h1>FoldCraft</h1>

This is the Novel Therapeutics working repository. The `baseline/` directory
contains experimental evaluation tools and recorded June 2026 results; these do
not establish an improvement in binding accuracy. See the
[integration status and known limitations](baseline/INTEGRATION.md).
The Colab links below still run the original upstream notebooks.

<h3>Fold-conditioned de novo binder design</h3>
FoldCraft enables fold-conditioning of binder structure, enabling design of binders with diverse folds like TIM-barrels, solenoid folds or Ig-like domains. 
Using VHH conditioned framework FoldCraft can succesfully design single domain nanobody binders against diverse tergets. The FoldCraft pipeline is described in this preprint

You can easily run FoldCraft on Google Colab using this <a href="https://colab.research.google.com/github/KhondamirRustamov/FoldCraft/blob/main/FoldCraft.ipynb">link</a>. Or use this <a href='https://colab.research.google.com/github/KhondamirRustamov/FoldCraft/blob/main/FoldCraft_VHH.ipynb'>notebook</a>, if you want to design VHH fold binders specifically

<br>
Paper: https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.abstract
<br>
<br>


![Fig1-1](https://github.com/user-attachments/assets/b7612207-be45-410d-aaff-fc2586ea765e)


<h2>Installation</h2>

First you need to install FoldCraft repository on your local machine:

`git clone https://github.com/Novel-Therapeutics/FoldCraft.git`

Then run code below to download all requirements, ColabDesign and AlphaFold2 weights

`bash install_foldcraft.sh --cuda '12.4' --pkg_manager 'conda'`

NOTE: AlphaFold3, which has been used in the manuscript for VHH design benchmarking, should be installed separately as described in official <a href='https://github.com/google-deepmind/alphafold3'>repository</a>

<h2>Running FoldCraft locally</h2>

To run FoldCraft locally you will need the following files:

```
--output_folder       ->      Folder to save the results
--binder_template     ->      Path to the binder template PDB file (required)
--target_template     ->      Path to the target template PDB file (required)
--target_hotspots     ->      Residue ranges for target hotspots, e.g., "14-30,80-81,90-102" (required)
```

Then you should activate you conda/mamba ebvionment with `conda activate FoldCraft` and run FoldCraft with following comand:
```
python FoldCraft.py \
      --output_folder design_1qys_pd_l1 \
      --binder_template 1qys.pdb \
      --target_template pd_l1.pdb \
      --target_hotspots '36-41,84-88,92-96' \
      --binder_hotspots '25-40,50-65' \
      --num_designs 40
```
Or run this code if you want to design nanobody fold-conditioned binders using VHH framework. In that case you don't need to specify the binder template pdb
```
python FoldCraft.py \
      --vhh \
      --output_folder design_vhh_pd_l1 \
      --target_template pd_l1.pdb \
      --target_hotspots '36-41,84-88,92-96' \
      --num_designs 40
```

You can also specify the ProteinMPNN optimization strategy, and ProteinMPNN weights to use, as well as other settings:
```
--vhh                 ->      Whether to use VHH framework to construct target cmap (all binder information would be ignored in that case)
--sample              ->      Whether to generate designs until the target number of successful designs is reached (default: False)
--target_success      ->      Target number of successful designs to generate - used only if --sample is enabled (default: 100)
--num_designs         ->      Number of design trajectories to generate - ignored if --sample is enabled (default: 1)
--binder_hotspots     ->      Residue ranges for binder, e.g. "14-30,80-81,90-102"
--binder_mask         ->      Residue ranges in the binder to mask (ignored during loss computation), e.g. "14-30"
--binder_chain        ->      Binder template chain (default: 'A')
--target_chain        ->      Target template chain (default: 'A')
--design_stages       ->      Number of each design stages in 3stage_design (default: 100,100,20)
--mpnn_weight         ->      PoteinMPNN weights to use: 'soluble', 'original' (default: 'soluble')
--redesign_method     ->      ProteinMPNN redesign strategy: 'full' or 'non-interface' (default: 'non-interface')
--mpnn_samples        ->      Number of sequences to sample with ProteinMPNN (default: 5)
--mpnn_backbone_noise ->      Backbone noise during sampling (default: 0.0)
--mpnn_sampling_temp  ->      Sampling temperature for amino acids 0.0-1.0 (default: 0.1)
--mpnn_save           ->      Whether to save MPNN sampled sequences
```

<h2>Credits</h2>

This repository uses code from:

* Sergey Ovchinnikov's ColabDesign (https://github.com/sokrypton/ColabDesign)

* Justas Dauparas's ProteinMPNN (https://github.com/dauparas/ProteinMPNN)

*   Martin Pacesa's BindCraft (https://github.com/martinpacesa/BindCraft)

### Correctness branch: CPU preflight and durable runs

See [CPU_FIXES.md](CPU_FIXES.md) for the implemented fixes, validation and remaining
GPU checks. Hotspot/mask inputs now use **original PDB residue numbers**, inclusive
at both endpoints; the driver records their mapping to model positions. Run your
normal command with `--preflight_only` to check inputs without GPU packages or
weights. Use a **new output directory** for each direct run. Success sampling is
bounded by `--max_trajectories` (default 1000); `--seed` enables replay, and
`--data_dir` selects the AlphaFold weight location.

New campaigns default to `runs/`; committed `baseline/` results are historical
reference data. GPU scorers use provenance sidecars and recompute legacy CSV-only
scores when invoked, so score copies of historical tables when preserving them.
The notebooks now call this checkout's CLI and require a prepared environment
for their inference cell.
