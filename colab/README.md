# Colab operator runbook

The scripts are designed for one named persistent session so weights can be
downloaded before the VM is stopped.

1. Create `image-restoration-project.tar.gz` locally without raw data/artifacts.
2. `colab new --gpu T4 -s kla-benchmark-2026`
3. Upload the source archive to `/content/image-restoration-project.tar.gz`.
4. Execute `prepare_remote.py` to install minimal dependencies and download the
   official training archive directly from its public Google Drive file ID.
5. Execute `run_remote_smoke.py` and download its archive.
6. Execute `run_remote_full.py` and download its archive.
7. Always stop the named session after downloading results.

For the project-specific successor model, run `run_remote_daf_smoke.py` first.
After it passes, `run_remote_daf_ablation.py` compares the original Restormer,
DAF-Restormer without synthetic degradation, and the complete DAF-Restormer at
the same 10-epoch/full-data budget.

The smoke profile uses 128 training and 64 validation samples for one epoch.
The full fixed-budget profile uses all 2,880 training and 320 validation samples
for three epochs per model. Neither should be presented as fully converged
competition training.
