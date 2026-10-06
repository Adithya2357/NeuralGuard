"""Checks on the deployment and packaging files: Dockerfile, pyproject.toml, CI workflow."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def dockerfile_instructions():
    """The Dockerfile's instructions, continuation lines joined, comments left out."""
    text = re.sub(r"\\\n", " ", (ROOT / "Dockerfile").read_text())
    return [line.strip() for line in text.splitlines() if line.strip() and line[0] != "#"]


def test_the_image_trains_the_model_neuralguard_train_does():
    # The compose detector and the CI smoke test run the image's model. It used to be
    # trained on a third less traffic than the default, with twice the false alarms.
    (run,) = [i for i in dockerfile_instructions() if re.match(r"RUN neuralguard train\b", i)]
    command = shlex.split(run.removeprefix("RUN").split("&&")[0])
    assert command[:2] == ["neuralguard", "train"]
    assert not {"--samples", "--trees", "--window", "--seed"} & set(command)


def test_the_image_can_be_stopped_while_its_command_is_starting():
    # PID 1 ignores signals without a handler; SIGINT always has Python's.
    assert "STOPSIGNAL SIGINT" in dockerfile_instructions()


def test_the_license_is_an_spdx_expression():
    # The table form and License classifiers are deprecated: builds with a future
    # setuptools would fail on them.
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    (setuptools,) = [r for r in config["build-system"]["requires"] if r.startswith("setuptools")]
    minimum = re.fullmatch(r"setuptools>=(\d+)(\.\d+)*", setuptools)
    assert minimum and int(minimum.group(1)) >= 77  # the first with license expressions


def test_jobs_sharing_a_pip_cache_install_the_same_packages():
    # setup-python's cache key is (OS, Python version, hash of cache-dependency-path):
    # nothing job-specific. The ruff-only lint job used to save the 3.12 cache first, so
    # the 3.12 test and security jobs downloaded every wheel on every run.
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    installs_by_key: dict[tuple[str, str], set[str]] = {}
    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        installs = "\n".join(s["run"] for s in steps if "pip install" in s.get("run", ""))
        matrix = job.get("strategy", {}).get("matrix", {})
        for step in steps:
            options = step.get("with", {})
            if not step.get("uses", "").startswith("actions/setup-python"):
                continue
            if options.get("cache") != "pip":
                continue
            version = str(options["python-version"])
            versions = matrix["python-version"] if "matrix." in version else [version]
            for each in versions:
                key = (str(each), str(options.get("cache-dependency-path")))
                installs_by_key.setdefault(key, set()).add(installs)
    assert installs_by_key  # the test jobs do cache
    for key, installs in installs_by_key.items():
        assert len(installs) == 1, f"pip cache {key} shared by different installs: {installs}"
