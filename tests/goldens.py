"""Shared goldens storage, keyed by platform where values are not portable.

A golden is a benchmark's output for a fixed input. Most of them travel: the
arithmetic benchmarks (bbob, pbo, graph, maxsat, lasso) give bit-identical
results on macOS/arm64 and in the linux/amd64 container, so they live under the
`any` section and one recording serves every machine.

Physics rollouts do not travel. Recorded on arm64 and replayed on amd64 at the
same seed, MuJoCo drifts from 2e-6 to a sign flip:

    mujoco-walker   4.309704043416025  ->  -1.7118133581203339

Those are stored per platform instead. When the running platform has no section,
the caller is expected to fall back to relative assertions -- same seed twice is
equal, different seeds differ -- which hold everywhere because they never compare
against a stored constant.

File shape:

    {
      "any":          {"bbob-sphere": 104.51646976},
      "linux-x86_64": {"mujoco-swimmer": -5.805574154153944},
      "darwin-arm64": {"mujoco-swimmer": -5.9325575192963615}
    }

Loaded by package conftests via sys.path, the way tests/grpc_harness.py is.
Standard library only, so it works in every package virtualenv.
"""
import json
import platform
import re
from pathlib import Path

# Values shared by every platform.
SHARED = "any"

# system-machine, e.g. darwin-arm64 or linux-x86_64. The system is part of the
# key, not just the machine: macOS-arm64 and linux-arm64 are different
# environments and must not share recorded values.
PLATFORM_KEY = f"{platform.system()}-{platform.machine()}".lower()

# What a valid section name looks like, for the Tier 0 schema check.
SECTION_PATTERN = re.compile(r"^[a-z0-9_]+-[a-z0-9_]+$")


def is_valid_section(name: str) -> bool:
    return name == SHARED or bool(SECTION_PATTERN.match(name))


def load(path: Path) -> dict:
    """Read a goldens file, or an empty mapping if it does not exist yet."""
    if not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def lookup(goldens: dict, name: str, platform_specific: bool = False):
    """The recorded value for `name`, or None if this platform has none.

    Platform-specific goldens resolve against this machine's section only --
    falling back to `any` would silently compare a rollout against another
    architecture's value, which is the bug this whole mechanism exists to avoid.
    """
    if platform_specific:
        return goldens.get(PLATFORM_KEY, {}).get(name)
    return goldens.get(SHARED, {}).get(name)


def record(path: Path, recorded: dict, platform_specific: bool = False) -> str:
    """Merge `recorded` into the right section and write the file back.

    Always merges. Overwriting meant `pytest -k sphere --update-goldens` silently
    dropped the other 55 goldens, which is how two packages' files could have
    drifted without anyone noticing.
    """
    path = Path(path)
    section = PLATFORM_KEY if platform_specific else SHARED

    goldens = load(path)
    goldens.setdefault(section, {}).update(recorded)
    path.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n")
    return section
