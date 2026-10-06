"""Create the isolated BELA environment, clone BELA, and fetch required assets."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path


BELA_REPOSITORY = "https://github.com/facebookresearch/BELA.git"
ASSETS = {
    "model_e2e.ckpt": "http://dl.fbaipublicfiles.com/bela/models/model_e2e.ckpt",
    "index.txt": "http://dl.fbaipublicfiles.com/bela/embeddings/index.txt",
    "embeddings.pt": "http://dl.fbaipublicfiles.com/bela/embeddings/embeddings.pt",
}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=base / ".venv-bela")
    parser.add_argument("--repo", type=Path, default=base / "external" / "BELA")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Directory for the large BELA model/index assets (default: REPO/models)",
    )
    parser.add_argument("--python-version", default="3.9")
    parser.add_argument("--skip-assets", action="store_true", help="Prepare code/dependencies only")
    return parser.parse_args()


def run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    rendered = [str(item) for item in command]
    print("\n>", subprocess.list2cmdline(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, check=True)


def validate_platform() -> None:
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "The official BELA/faiss-gpu stack requires Linux x86-64. "
            f"Detected {platform.system()} {platform.machine()}."
        )


def find_uv() -> Path | None:
    executable = shutil.which("uv")
    candidates = [
        Path(executable) if executable else None,
        Path(sys.executable).resolve().parent / "uv",
        Path(sysconfig.get_path("scripts")) / "uv",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    return None


def ensure_uv() -> Path:
    executable = find_uv()
    if executable:
        return executable
    run([sys.executable, "-m", "pip", "install", "uv>=0.8,<1"])
    executable = find_uv()
    if executable is None:
        raise FileNotFoundError("uv was installed but its executable could not be located")
    return executable


def environment_python(environment: Path) -> Path:
    return environment / "bin" / "python"


def ensure_environment(environment: Path, python_version: str) -> Path:
    bela_python = environment_python(environment)
    if not bela_python.is_file():
        uv = ensure_uv()
        run([uv, "python", "install", python_version])
        environment.parent.mkdir(parents=True, exist_ok=True)
        run([uv, "venv", "--seed", "--python", python_version, environment])
    probe = subprocess.run(
        [
            str(bela_python),
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({'version': "
                "f'{sys.version_info.major}.{sys.version_info.minor}', "
                "'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    details = json.loads(probe.stdout)
    detected = str(details["version"])
    if detected != python_version:
        raise RuntimeError(
            f"Existing BELA environment uses Python {detected}, expected {python_version}: {environment}"
        )
    detected_prefix = Path(str(details["prefix"])).resolve()
    expected_prefix = environment.resolve()
    if detected_prefix != expected_prefix:
        raise RuntimeError(
            "BELA interpreter is not running inside the requested virtual environment: "
            f"sys.prefix={detected_prefix}, expected={expected_prefix}. "
            "Recreate the environment with uv venv --seed."
        )
    if Path(str(details["base_prefix"])).resolve() == detected_prefix:
        raise RuntimeError(f"BELA path is not an isolated virtual environment: {environment}")
    # Do not call resolve() on bin/python: uv virtual environments normally use
    # a symlink to the managed base interpreter. Executing the symlink path is
    # what makes CPython discover pyvenv.cfg and activate the isolated prefix.
    return bela_python


def ensure_repository(repo: Path) -> None:
    if (repo / ".git").is_dir() and (repo / "bela").is_dir():
        print(f"Reusing BELA repository: {repo}")
        return
    if repo.exists():
        raise RuntimeError(
            f"BELA repository target exists but is not a valid clone: {repo}"
        )
    repo.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if not git:
        raise FileNotFoundError("Git is required to clone BELA")
    run([git, "clone", BELA_REPOSITORY, repo])


def requirements_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_dependencies(bela_python: Path, repo: Path, requirements: Path) -> None:
    run(
        [
            bela_python,
            "-m",
            "pip",
            "install",
            "pip<24.1",
            "setuptools==80.9.0",
        ]
    )
    run([bela_python, "-m", "pip", "install", "-r", requirements])
    run([bela_python, "-m", "pip", "install", "-e", repo])
    run(
        [
            bela_python,
            "-c",
            (
                "from importlib.metadata import version; "
                "assert version('setuptools') == '80.9.0', version('setuptools'); "
                "assert version('six') == '1.16.0', version('six'); "
                "import pkg_resources, six; "
                "from bela.datamodule.joint_el_datamodule import JointELDataModule; "
                "from bela.evaluation.model_eval import ModelEval; "
                "print('BELA runtime imports: OK; setuptools=' + version('setuptools') "
                "+ '; six=' + version('six'))"
            ),
        ]
    )


def migrate_legacy_assets(repo: Path, assets_dir: Path) -> None:
    """Move reusable assets/partials from REPO/models to a separate filesystem."""

    legacy_dir = (repo / "models").resolve()
    destination_dir = assets_dir.resolve()
    if legacy_dir == destination_dir or not legacy_dir.is_dir():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in ASSETS:
        for suffix in ("", ".part"):
            source = legacy_dir / f"{filename}{suffix}"
            destination = destination_dir / source.name
            if not source.is_file():
                continue
            if destination.exists():
                print(
                    "Keeping legacy BELA file because the destination already exists: "
                    f"{source}",
                    flush=True,
                )
                continue
            print(f"Moving reusable BELA download: {source} -> {destination}", flush=True)
            shutil.move(str(source), str(destination))


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value else None


def download_with_resume(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    expected_size = remote_size(url)
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        print(f"Reusing BELA asset: {target}")
        return
    if target.exists():
        if partial.exists() and partial.stat().st_size >= target.stat().st_size:
            target.unlink()
        else:
            target.replace(partial)
    if partial.is_file() and expected_size is not None and partial.stat().st_size == expected_size:
        partial.replace(target)
        return

    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if not append:
            offset = 0
        response_size = response.headers.get("Content-Length")
        total = offset + int(response_size) if response_size else expected_size
        mode = "ab" if append else "wb"
        downloaded = offset
        next_report = downloaded + 256 * 1024 * 1024
        try:
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total:
                            print(f"{target.name}: {downloaded / total:.1%}", flush=True)
                        else:
                            print(f"{target.name}: {downloaded / (1024**3):.1f} GiB", flush=True)
                        next_report = downloaded + 256 * 1024 * 1024
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            if error.errno in {errno.EDQUOT, errno.ENOSPC}:
                partial_size = partial.stat().st_size if partial.exists() else 0
                raise RuntimeError(
                    "Insufficient disk space or user quota for BELA assets. "
                    f"Partial download retained at {partial} "
                    f"({partial_size / (1024**3):.2f} GiB). Re-run setup with "
                    "--assets-dir on a filesystem with at least 21 GiB available."
                ) from error
            raise
    final_size = partial.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise RuntimeError(
            f"Incomplete BELA asset {target.name}: {final_size} != {expected_size} bytes. "
            "Re-run the setup cell to resume it."
        )
    partial.replace(target)
    print(f"Downloaded {target} ({final_size / (1024**3):.2f} GiB)")


def main() -> None:
    args = parse_args()
    validate_platform()
    base = Path(__file__).resolve().parent
    environment = args.environment.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    assets_dir = (
        args.assets_dir.expanduser().resolve()
        if args.assets_dir is not None
        else repo / "models"
    )
    requirements = base / "requirements-bela.txt"
    bela_python = ensure_environment(environment, args.python_version)
    ensure_repository(repo)
    install_dependencies(bela_python, repo, requirements)
    if not args.skip_assets:
        migrate_legacy_assets(repo, assets_dir)
        for filename, url in ASSETS.items():
            download_with_resume(url, assets_dir / filename)
    marker = environment / "j2ctst_bela_setup.json"
    marker.write_text(
        json.dumps(
            {
                "python": str(bela_python),
                "python_version": args.python_version,
                "repository": str(repo),
                "assets_directory": str(assets_dir),
                "requirements_sha256": requirements_sha256(requirements),
                "assets": [] if args.skip_assets else sorted(ASSETS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"BELA setup complete. Interpreter: {bela_python}")


if __name__ == "__main__":
    main()
