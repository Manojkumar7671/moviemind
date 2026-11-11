"""tests/test_docker_build.py

Verifies that:
1. Dockerfile does not reference requirements-train.txt (torch excluded from runtime image).
2. The git-tracked file set does not include artifacts/global_catalog.parquet or the three derived artifacts.
3. .dockerignore lists the four generated artifacts.
4. Dockerfile HEALTHCHECK --start-period is at least 60s and non-runtime files (tests/, docs/, pytest.ini) are not copied.
"""

from pathlib import Path
import re
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_dockerfile_does_not_contain_requirements_train():
    dockerfile_path = ROOT_DIR / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile not found"
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "requirements-train.txt" not in content, (
        "Dockerfile must NOT contain or copy 'requirements-train.txt' (torch must remain excluded from slim runtime image)"
    )


def test_tracked_files_exclude_world_catalog_and_derived_artifacts():
    out = subprocess.check_output(["git", "ls-files", "artifacts"], cwd=str(ROOT_DIR)).decode("utf-8")
    tracked_files = [line.strip() for line in out.strip().splitlines() if line.strip()]

    prohibited_files = {
        "artifacts/global_catalog.parquet",
        "artifacts/global_tfidf.npz",
        "artifacts/global_vectorizer.pkl",
        "artifacts/global_lookups.json",
    }

    for prohibited in prohibited_files:
        assert prohibited not in tracked_files, (
            f"Prohibited file '{prohibited}' is tracked in git! It must be git-ignored and built during Docker build."
        )

    # Ensure required wikidata_cache is tracked
    assert "artifacts/wikidata_cache.json" in tracked_files, (
        "artifacts/wikidata_cache.json must be tracked in git for catalog build."
    )


def test_dockerignore_lists_generated_artifacts():
    dockerignore_path = ROOT_DIR / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore not found"
    content = dockerignore_path.read_text(encoding="utf-8")
    ignored_patterns = {line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")}

    required_artifacts = [
        "artifacts/global_catalog.parquet",
        "artifacts/global_tfidf.npz",
        "artifacts/global_vectorizer.pkl",
        "artifacts/global_lookups.json",
    ]

    for artifact in required_artifacts:
        assert artifact in ignored_patterns, (
            f".dockerignore must explicitly ignore '{artifact}'"
        )


def test_dockerfile_healthcheck_start_period_and_runtime_exclusions():
    dockerfile_path = ROOT_DIR / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile not found"
    content = dockerfile_path.read_text(encoding="utf-8")

    # Verify HEALTHCHECK --start-period is at least 60s
    match = re.search(r"--start-period=(\d+)s", content)
    assert match is not None, "Dockerfile HEALTHCHECK must specify --start-period=<N>s"
    start_period = int(match.group(1))
    assert start_period >= 60, (
        f"HEALTHCHECK --start-period must be at least 60s, got {start_period}s"
    )

    # Verify non-runtime directories/files (tests/, docs/, pytest.ini) are not copied
    assert "COPY tests" not in content, "Dockerfile should not copy tests/ into runtime image"
    assert "COPY docs" not in content, "Dockerfile should not copy docs/ into runtime image"
    assert "COPY pytest.ini" not in content, "Dockerfile should not copy pytest.ini into runtime image"
