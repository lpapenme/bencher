"""Guards benchmark-registry.json against drift from the services.

The registry is the only thing the front door advertises (BencherServer/main.py
reads it to build one stub per (host, port)), but nothing at runtime validates
it: `dimensions` and `type` are read by no runtime code, so a wrong entry would
otherwise surface only as an AssertionError inside a family service at RPC time.

Each service now declares its own benchmarks, dimensionality and type, so these
tests can cross-check the two against each other statically.
"""
import re

import pytest

from conftest import (REPO_ROOT, declared_specs, dispatch_names, registry_by_port, rel)

# Ports the front door knows how to override via env vars
# (BencherServer/src/bencherserver/main.py:18-38).
KNOWN_PORTS = {50053, 50054, 50055, 50056, 50057, 50058, 50059, 50060}

VALID_TYPES = {
    "purely_continuous", "purely_binary", "purely_categorical",
    "purely_ordinal_real", "purely_ordinal_int", "purely_integer", "mixed",
}

# port -> (service main.py, dispatch variables to read)
# Only families whose names are statically enumerable. IOH (prefix matching) and
# BO4Mob (regex) are checked separately below.
STATIC_FAMILIES = {
    50053: ("LassoBenchmarks/src/lassobenchmarks/main.py", {"BENCHMARKS"}),
    # Refactored to the declarative BENCHMARKS mapping. As each remaining
    # service follows, switch its entry here; once all nine have, this table
    # collapses to a single variable name and dispatch_names' fallbacks for
    # `in [...]` / `== "..."` comparisons become dead code.
    50054: ("NoDependencyBenchmark/src/nodependencybenchmark/main.py", {"BENCHMARKS"}),
    50055: ("MaxSATBenchmarks/src/maxsatbenchmarks/main.py", {"BENCHMARKS"}),
    50056: ("EboBenchmarks/src/ebobenchmarks/main.py", {"BENCHMARKS"}),
    50057: ("MujocoBenchmarks/src/mujocobenchmarks/main.py", {"BENCHMARKS"}),
    50058: ("SVMBenchmarks/src/svmbenchmarks/main.py", {"BENCHMARKS"}),
}

IOH_PORT = 50059
IOH_PREFIXES = ("bbob-", "pbo-", "graph-")
BO4MOB_PORT = 50060
BO4MOB_MAIN = "BO4MobBenchmark/src/bo4mobbenchmark/main.py"


def test_registry_entries_have_the_expected_schema(registry):
    bad = []
    for name, props in registry.items():
        if set(props) != {"port", "dimensions", "type"}:
            bad.append(f"{name}: unexpected keys {sorted(props)}")
        elif props["port"] not in KNOWN_PORTS:
            bad.append(f"{name}: port {props['port']} has no env-var override mapping")
        elif props["type"] not in VALID_TYPES:
            bad.append(f"{name}: unknown type {props['type']!r}")
        elif not (props["dimensions"] is None or
                  (isinstance(props["dimensions"], int) and props["dimensions"] > 0)):
            bad.append(f"{name}: dimensions must be a positive int or null, "
                       f"got {props['dimensions']!r}")
    assert not bad, "malformed registry entries:\n" + "\n".join(bad)


@pytest.mark.parametrize("port", sorted(STATIC_FAMILIES), ids=lambda p: str(p))
def test_registry_matches_what_the_service_dispatches(port, registry):
    """Every advertised name must be served, and every served name advertised."""
    main_rel, variables = STATIC_FAMILIES[port]
    served = dispatch_names(REPO_ROOT / main_rel, variables)
    listed = registry_by_port(registry).get(port, set())

    advertised_but_unserved = listed - served
    served_but_unadvertised = served - listed
    assert not advertised_but_unserved, (
        f"{main_rel} does not serve {sorted(advertised_but_unserved)}, "
        f"but the registry advertises them on port {port}")
    assert not served_but_unadvertised, (
        f"{main_rel} serves {sorted(served_but_unadvertised)}, "
        f"but they are missing from the registry (clients cannot reach them)")


def test_ioh_names_use_a_known_prefix(registry):
    """IOH dispatches by prefix at runtime, so only the prefix is checkable here.

    The exact name->problem check lives in IOHBenchmarks' own tests, where `ioh`
    is installed.
    """
    listed = registry_by_port(registry).get(IOH_PORT, set())
    assert listed, "no IOH benchmarks registered"
    bad = sorted(n for n in listed if not n.startswith(IOH_PREFIXES))
    assert not bad, f"IOH names must start with one of {IOH_PREFIXES}: {bad}"


def test_bo4mob_names_match_the_service_networks(registry):
    """BO4Mob templates 420 names from 5 networks, so check the networks.

    Every advertised name must belong to a declared network and carry that
    network's dimensionality (the OD-pair count of its CSV template).
    """
    networks = declared_specs(REPO_ROOT / BO4MOB_MAIN, "NETWORKS")
    assert networks, f"{BO4MOB_MAIN} declares no NETWORKS"

    listed = {n: registry[n] for n in registry_by_port(registry).get(BO4MOB_PORT, set())}
    assert listed, "no BO4Mob benchmarks registered"

    problems = []
    for name, entry in sorted(listed.items()):
        network = name.split("_")[0]
        if network not in networks:
            problems.append(f"{name}: network {network!r} is not declared by the service")
            continue
        expected = networks[network].get("dimensions")
        if expected is not None and entry["dimensions"] != expected:
            problems.append(f"{name}: registry says {entry['dimensions']} dims, "
                            f"the {network} template has {expected}")
    assert not problems, "BO4Mob registry drift:\n" + "\n".join(problems[:10])


def test_every_bo4mob_network_is_represented_in_the_registry(registry):
    """A declared network with no registry entries is unreachable."""
    networks = declared_specs(REPO_ROOT / BO4MOB_MAIN, "NETWORKS")
    listed = registry_by_port(registry).get(BO4MOB_PORT, set())
    missing = sorted(n for n in networks
                     if not any(name.startswith(f"{n}_") for name in listed))
    assert not missing, f"networks the service serves but the registry omits: {missing}"


def test_every_known_port_is_represented(registry):
    """A service with no registry entries is unreachable through the front door."""
    present = set(registry_by_port(registry))
    missing = KNOWN_PORTS - present
    assert not missing, f"ports with no registered benchmarks: {sorted(missing)}"


def test_readme_table_agrees_with_the_registry(registry):
    """README's benchmark table duplicates the registry; keep the two in step.

    Only names and dimensions are cross-checked -- the table also carries source
    and noisiness columns that have no registry counterpart. Rows whose name ends
    in `_*` are templated families (the BO4Mob networks): every registry entry
    with that prefix must carry the stated dimensionality.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    rows = re.findall(r"^\|\s*([A-Za-z0-9_.*-]+)\s*\|\s*(\d+|any)\s*\|",
                      readme, re.MULTILINE)
    assert rows, "no parseable benchmark table in README.md"

    mismatches = []
    for name, dims in rows:
        expected_any = dims == "any"
        if name.endswith("_*"):
            prefix = name[:-1]
            matching = {n: p for n, p in registry.items() if n.startswith(prefix)}
            if not matching:
                mismatches.append(f"README lists {name!r}, but no registry entry "
                                  f"starts with {prefix!r}")
                continue
            wrong = {n: p["dimensions"] for n, p in matching.items()
                     if p["dimensions"] != (None if expected_any else int(dims))}
            if wrong:
                sample = sorted(wrong.items())[:3]
                mismatches.append(f"{name}: README says {dims} dims, but "
                                  f"{len(wrong)} registry entries differ, e.g. {sample}")
            continue

        if name not in registry:
            mismatches.append(f"README lists {name!r}, which is not in the registry")
            continue
        actual = registry[name]["dimensions"]
        if expected_any and actual is not None:
            mismatches.append(f"{name}: README says any-dimensional, registry says {actual}")
        elif not expected_any and actual != int(dims):
            mismatches.append(f"{name}: README says {dims} dims, registry says {actual}")

    assert not mismatches, "README/registry drift:\n" + "\n".join(mismatches)


# Services that declare dimensionality and type alongside their dispatch. The
# registry records both for every benchmark but no runtime code reads either, so
# without this check they are free to drift from what the service actually wants.
DECLARING_FAMILIES = {
    port: main_rel for port, (main_rel, variables) in STATIC_FAMILIES.items()
    if "BENCHMARKS" in variables
}


@pytest.mark.parametrize("port", sorted(DECLARING_FAMILIES), ids=lambda p: str(p))
def test_declared_dimensions_and_type_match_the_registry(port, registry):
    specs = declared_specs(REPO_ROOT / DECLARING_FAMILIES[port])
    assert specs, f"{DECLARING_FAMILIES[port]} declares no BENCHMARKS specs"

    mismatches = []
    for name, fields in specs.items():
        entry = registry.get(name)
        if entry is None:
            mismatches.append(f"{name}: declared by the service, absent from the registry")
            continue
        for field in ("dimensions", "type"):
            if field in fields and fields[field] != entry[field]:
                mismatches.append(
                    f"{name}: service declares {field}={fields[field]!r}, "
                    f"registry says {entry[field]!r}")
    assert not mismatches, "service/registry drift:\n" + "\n".join(mismatches)
