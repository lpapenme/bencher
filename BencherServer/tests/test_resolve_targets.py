"""Unit tests for the registry -> (host, port) resolution.

This logic used to live inline in serve(), immediately above a blocking server
start, so nothing could exercise it -- yet it decides which service every one of
the 500 registered benchmarks is routed to, and it honours a dozen env vars.
"""
import pytest

from bencherserver.main import load_registry, resolve_targets


def test_entries_group_by_target():
    registry = {
        "a": {"port": 50053, "dimensions": 1, "type": "purely_continuous"},
        "b": {"port": 50053, "dimensions": 2, "type": "purely_continuous"},
        "c": {"port": 50054, "dimensions": 3, "type": "purely_continuous"},
    }
    targets = resolve_targets(registry, env={})
    assert targets == {("localhost", 50053): ["a", "b"], ("localhost", 50054): ["c"]}


def test_port_env_var_overrides_the_registry():
    registry = {"lasso-dna": {"port": 50053, "dimensions": 180, "type": "purely_continuous"}}
    targets = resolve_targets(registry, env={"BENCHER_LASSO_PORT": "60053"})
    assert targets == {("localhost", 60053): ["lasso-dna"]}


def test_host_env_var_overrides_the_default():
    registry = {"svm": {"port": 50058, "dimensions": 388, "type": "purely_continuous"}}
    targets = resolve_targets(registry, env={"BENCHER_SVM_HOST": "svm-box"})
    assert targets == {("svm-box", 50058): ["svm"]}


def test_host_and_port_overrides_combine():
    """The host lookup keys off the *original* port, after port was reassigned."""
    registry = {"svm": {"port": 50058, "dimensions": 388, "type": "purely_continuous"}}
    targets = resolve_targets(
        registry, env={"BENCHER_SVM_HOST": "svm-box", "BENCHER_SVM_PORT": "60058"})
    assert targets == {("svm-box", 60058): ["svm"]}


def test_only_the_first_host_of_a_list_is_used():
    """Documents a real limitation rather than asserting it is desirable."""
    registry = {"svm": {"port": 50058, "dimensions": 388, "type": "purely_continuous"}}
    targets = resolve_targets(registry, env={"BENCHER_SVM_HOST": "first,second"})
    assert targets == {("first", 50058): ["svm"]}


def test_unknown_port_gets_no_override():
    registry = {"x": {"port": 59999, "dimensions": 1, "type": "purely_continuous"}}
    assert resolve_targets(registry, env={"BENCHER_LASSO_PORT": "1"}) == \
           {("localhost", 59999): ["x"]}


def test_the_real_registry_resolves_to_the_expected_targets():
    """Every registered benchmark must land on exactly one target."""
    registry = load_registry()
    targets = resolve_targets(registry, env={})
    routed = [name for names in targets.values() for name in names]
    assert sorted(routed) == sorted(registry)
    assert len(routed) == len(set(routed)), "a benchmark was routed to two targets"
