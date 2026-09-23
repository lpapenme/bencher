#!/usr/bin/env python3
"""Emit the CI test matrix, derived from the repo rather than hand-listed.

A package is any top-level directory with a pyproject.toml -- the same rule
tests/conftest.py uses. Each declares where its tests can run:

    [tool.bencher.ci]
    tier = "runner"      # a plain GitHub runner (the default)
    tier = "container"   # needs the built image: MuJoCo binaries, SUMO, ...

Writing the tier next to the package keeps it visible in the same diff that adds
a heavy dependency, and defaulting to "runner" means a new package shows up as an
extra CI leg rather than silently going untested.

Prints `runner=<json>` and `container=<json>` for $GITHUB_OUTPUT.
"""
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIER = "runner"
VALID_TIERS = {"runner", "container"}


def discover() -> dict[str, list[dict]]:
    matrix: dict[str, list[dict]] = {tier: [] for tier in VALID_TIERS}

    for pyproject in sorted(REPO_ROOT.glob("*/pyproject.toml")):
        package = pyproject.parent
        if not (package / "tests").is_dir():
            continue

        config = tomllib.loads(pyproject.read_text())
        tier = (config.get("tool", {}).get("bencher", {})
                .get("ci", {}).get("tier", DEFAULT_TIER))
        if tier not in VALID_TIERS:
            print(f"::error file={pyproject}::unknown [tool.bencher.ci] tier "
                  f"{tier!r}; expected one of {sorted(VALID_TIERS)}", file=sys.stderr)
            raise SystemExit(1)

        version_file = package / ".python-version"
        matrix[tier].append({
            "package": package.name,
            "python": version_file.read_text().strip() if version_file.is_file() else "",
        })

    return matrix


def main() -> None:
    matrix = discover()
    for tier, entries in sorted(matrix.items()):
        print(f"{tier}={json.dumps(entries)}")
    names = {t: [e["package"] for e in v] for t, v in matrix.items()}
    print(f"::notice::test matrix -> {names}", file=sys.stderr)


if __name__ == "__main__":
    main()
