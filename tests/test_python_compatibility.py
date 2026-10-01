"""Guards each package's source against the Python version it actually pins.

The packages run on three different interpreters (3.8, 3.10, 3.11) because their
upstream libraries are mutually incompatible. It is easy to write something that
is fine on 3.11 and fatal on 3.8 -- and the failure mode is nasty: the service
dies at container start, so it surfaces only after a full image build.

That happened: `seed: int | None = None` (PEP 604) was added to every service,
which raises TypeError at definition time on 3.8. EboBenchmarks and
MujocoBenchmarks both stopped starting, and it took a ~20 minute build to find.
These checks take milliseconds.
"""
import ast
import tomllib

import pytest

from conftest import package_dirs, parse, rel

# (major, minor) below which a construct is unavailable.
PEP_604_MIN = (3, 10)   # X | Y in annotations
MATCH_MIN = (3, 10)     # structural pattern matching
WALRUS_MIN = (3, 8)

PACKAGES = package_dirs()
PACKAGE_IDS = [p.name for p in PACKAGES]


def _pinned_version(pkg) -> tuple[int, int] | None:
    version_file = pkg / ".python-version"
    if not version_file.is_file():
        return None
    parts = version_file.read_text().strip().split(".")
    return (int(parts[0]), int(parts[1]))


def _source_files(pkg):
    return sorted(pkg.glob("src/**/*.py"))


def _defers_annotations(tree: ast.Module) -> bool:
    """True if `from __future__ import annotations` is in effect."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _annotation_nodes(tree: ast.Module):
    """Every annotation expression in the module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                        args.vararg, args.kwarg]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_annotations_are_valid_for_the_pinned_interpreter(pkg):
    """PEP 604 (`int | None`) needs 3.10, or deferred annotations.

    Unlike a syntax error this parses fine everywhere -- it only explodes when
    the annotation is *evaluated*, at import time, on the older interpreter.
    """
    version = _pinned_version(pkg)
    if version is None or version >= PEP_604_MIN:
        pytest.skip(f"{pkg.name} pins {version}, which supports PEP 604")

    offenders = []
    for path in _source_files(pkg):
        tree = parse(path)
        if _defers_annotations(tree):
            continue
        for annotation in _annotation_nodes(tree):
            for node in ast.walk(annotation):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                    offenders.append(f"{rel(path)}:{node.lineno}")
    assert not offenders, (
        f"{pkg.name} pins Python {version[0]}.{version[1]}, where `X | Y` "
        f"annotations raise TypeError at import. Add "
        f"`from __future__ import annotations` or use typing.Optional:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_no_syntax_newer_than_the_pinned_interpreter(pkg):
    """Catches genuinely newer *syntax*, e.g. a match statement on 3.8.

    ast.parse(feature_version=...) rejects constructs the target cannot parse.
    """
    version = _pinned_version(pkg)
    if version is None:
        pytest.skip(f"{pkg.name} has no .python-version")

    failures = []
    for path in _source_files(pkg):
        try:
            ast.parse(path.read_text(), filename=str(path), feature_version=version)
        except SyntaxError as exc:
            failures.append(f"{rel(path)}:{exc.lineno}: {exc.msg}")
    assert not failures, (
        f"{pkg.name} pins Python {version[0]}.{version[1]} but its source uses "
        f"newer syntax:\n  " + "\n  ".join(failures))
