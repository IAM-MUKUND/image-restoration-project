"""Compare Restormer and DAF-Restormer variants at a matched 10-epoch budget."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
ROOT = PROJECT / "artifacts" / "daf-ablation"
ARCHIVE = Path("/content/kla-daf-ablation.tar.gz")


def run(label: str, model: str, extra: list[str]) -> None:
    output = ROOT / label
    command = [
        sys.executable,
        "benchmark_all.py",
        "--models",
        model,
        "--data-root",
        "/content/kla-data/extracted/train",
        "--output-dir",
        str(output),
        "--epochs",
        "10",
        "--batch-size",
        "8",
        "--num-workers",
        "2",
        *extra,
    ]
    subprocess.check_call(command, cwd=PROJECT)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    runs = [
        ("restormer-10ep", "restormer", ["--auxiliary-weight", "0", "--uncertainty-weight", "0"]),
        (
            "daf-no-synthetic-10ep",
            "daf_restormer",
            ["--frequency-weight", "0.03", "--gradient-weight", "0.02"],
        ),
        (
            "daf-full-10ep",
            "daf_restormer",
            [
                "--frequency-weight",
                "0.03",
                "--gradient-weight",
                "0.02",
                "--synthetic-probability",
                "0.35",
            ],
        ),
    ]
    for label, model, extra in runs:
        run(label, model, extra)

    comparison = {}
    for label, model, _ in runs:
        summary_path = ROOT / label / "benchmark_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        comparison[label] = summary["results"][model]
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(ROOT, arcname="daf-ablation")
    print(f"DAF_ABLATION_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
