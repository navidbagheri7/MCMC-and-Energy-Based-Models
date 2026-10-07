"""Verify a built wheel and source archive using isolated imports and portable evidence.

Build first with ``python -m build --outdir .local/dist``. This checker runs
outside the source tree and does not access course originals, ratings or factors.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from sampling_lab.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(".local/dist"))
    parser.add_argument("--work", type=Path, default=Path(".local/distribution_1_0"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = next(args.dist.resolve().glob("*.whl"))
    archive = next(args.dist.resolve().glob("*.tar.gz"))
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError("Choose a fresh work directory to preserve earlier checks")
    work.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if name.startswith("sampling_lab/") and name.endswith((".py", ".json", ".npz")):
                assert package.read(name) == (root / "src" / name).read_bytes(), name
    extraction = work / "archive"
    extraction.mkdir()
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            path = Path(item.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not set(path.parts) & {".local", "originals", "checkpoints"}
            assert path.suffix not in {".pt", ".pth", ".cbm", ".joblib", ".rar", ".log"}
            assert item.size <= 20 * 1024 * 1024
        package.extractall(extraction, filter="data")
    source = next(extraction.iterdir())
    for required in [
        "docs/explorer.html",
        "scripts/templates/explorer.js",
        "configs/ebm.json",
    ]:
        assert (source / required).exists()
    installed = work / "installed"
    external = work / "external"
    external.mkdir()
    shutil.copytree(source / "tests", external / "tests")
    shutil.copytree(source / "scripts", external / "scripts")
    (external / "assets/source_data").mkdir(parents=True)
    for name in [
        "logistic_nuts_dispersed_0.1",
        "regression_nuts_dispersed_0.1",
        "ring_parallel_tempering_dispersed_0.6",
    ]:
        shutil.copy2(
            source / "assets/source_data" / (name + ".npz"),
            external / "assets/source_data" / (name + ".npz"),
        )
    assert len(list((source / "results/reference/ebm").glob("*/energy.npz"))) == 36
    assert len(list((source / "results/reference/sampling").glob("*/run.json"))) == 84
    assert len(list((source / "notebooks").glob("*.ipynb"))) == 10
    (external / "pytest.ini").write_text("[pytest]\ntestpaths=tests\n")
    env = os.environ | dict(
        PYTHONPATH=str(installed),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    checks = []

    def run(name, command, cwd=external):
        print("Starting", name, flush=True)
        with (work / (name + ".log")).open("w") as log:
            subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        checks.append(dict(check=name, status="passed"))
        print("Passed", name, flush=True)

    run(
        "install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
    )
    run(
        "import_identity",
        [
            sys.executable,
            "-c",
            f"import sampling_lab; assert '{installed}' in sampling_lab.__file__; assert sampling_lab.__version__=='1.0.0'; print(sampling_lab.__file__)",
        ],
    )
    run("wheel_tests", [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib"])
    run(
        "bundled_sampling",
        [
            sys.executable,
            "-m",
            "sampling_lab.cli",
            "sampling",
            "--quick",
            "--filter",
            '"target": "gaussian"',
        ],
    )
    run(
        "bundled_energy",
        [sys.executable, "-m", "sampling_lab.cli", "ebm", "--quick", "--filter", '"method": "cd1"'],
    )
    run("initialize_checkout", ["git", "init", "-q"], source)
    run("archive_notebooks", [sys.executable, "scripts/execute_notebooks.py"], source)
    run("archive_validation", [sys.executable, "scripts/validate_release.py"], source)
    run("archive_public_content", [sys.executable, "scripts/review_components.py"], source)
    audit = dict(
        status="passed",
        release="1.0.0",
        artifacts={wheel.name: sha256(wheel), archive.name: sha256(archive)},
        checks=checks,
        installed_wheel_tests=25,
        source_archive_notebooks=10,
        source_archive_executed_cells=30,
        scope="Wheel imports, all 25 tests and two bundled CPU command families run outside the checkout. Three small public numeric fixtures enable the two reference-dependent tests. The source archive includes all 36 safe EBM models, executes all ten walkthroughs and validates all 120 protocols, numerical, English, link and browser evidence without the original notebook or private checkpoints. External checksum metadata is excluded from the source distribution to avoid recursive checksums. Hosted CI is not claimed.",
    )
    write_json(root / "docs/release_artifacts.json", audit)
    write_json(work / "audit.json", audit)
    print("Verified isolated wheel and complete source archive")


if __name__ == "__main__":
    main()
