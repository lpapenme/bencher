"""Guards the Dockerfile's per-package COPY lists and the build context.

The Dockerfile copies each package's pyproject.toml, uv.lock and .python-version
explicitly rather than by glob, because a glob would flatten nine identically
named files into one. Explicit lists drift when a package is added, so these
tests keep them honest -- a new benchmark that is not wired into the Dockerfile
would otherwise build an image silently missing that service.
"""
import re

import pytest

from conftest import REPO_ROOT, package_dirs

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text()


def _run_instruction_blocks():
    """Return ``(line_number, content)`` for executable Dockerfile RUN instructions."""
    blocks = []
    block = []
    for line_number, line in enumerate(DOCKERFILE.splitlines()):
        if block:
            block.append(line)
            if not line.rstrip().endswith("\\"):
                blocks.append((
                    line_number - len(block) + 1,
                    "\n".join(
                        instruction for instruction in block
                        if not instruction.lstrip().startswith("#")),
                ))
                block = []
        elif line.startswith("RUN "):
            block = [line]
            if not line.rstrip().endswith("\\"):
                blocks.append((line_number, line))
                block = []
    return blocks


def _without_inline_shell_comments(text):
    """Remove unquoted shell comments without treating quoted hashes as comments."""
    lines = []
    for line in text.splitlines(keepends=True):
        quote = None
        escaped = False
        for index, char in enumerate(line):
            if escaped:
                escaped = False
            elif char == "\\" and quote != "'":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == "#" and (
                index == 0 or line[index - 1].isspace() or line[index - 1] in ";&|()"
            ):
                line = line[:index]
                break
        lines.append(line)
    return "".join(lines)


def _mask_quoted_shell_control_operators(text):
    """Prevent quoted shell control operators from becoming command boundaries."""
    masked = []
    quote = None
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
            elif char in ";&|()\n":
                char = "\0"
        elif char in "'\"":
            quote = char
        masked.append(char)
    return "".join(masked)


def _has_shell_command(text, command):
    """Check for a command at a shell command boundary, not inside `echo` text."""
    text = _mask_quoted_shell_control_operators(
        _without_inline_shell_comments(text))
    boundary = (
        r"(?:^[ \t]*(?:RUN(?:[ \t]+--\S+)*[ \t]+)?"
        r"(?:\\[ \t]*\n[ \t]*)?"
        r"|(?:&&|;)[ \t]*(?:\\[ \t]*\n[ \t]*)?"
        r"|(?<!\\)\n[ \t]*)"
    )
    assignments = r"(?:[A-Za-z_]\w*=(?:[^\s;]+|'[^']*'|\"[^\"]*\")[ \t]+)*"
    return re.search(boundary + assignments + command, text, re.M)


PACKAGES = package_dirs()
PACKAGE_IDS = [p.name for p in PACKAGES]
DEPENDENCY_ONLY_SYNC = (
    r"uv[ \t]+sync\b(?:[^;&\n&|]|\\[ \t]*\n)*--no-install-project\b"
)


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_python_version_is_copied_for_every_package(pkg):
    """Without this the interpreter layer would miss a package's Python."""
    assert re.search(rf"^COPY\s+{re.escape(pkg.name)}/\.python-version\s", DOCKERFILE, re.M), (
        f"Dockerfile never copies {pkg.name}/.python-version; add it to the "
        f"interpreters stage")


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_dependency_files_are_copied_for_every_package(pkg):
    """The dependency layer must see this package's pyproject.toml and uv.lock."""
    line = re.search(rf"^COPY\s+{re.escape(pkg.name)}/pyproject\.toml\s+.*$",
                     DOCKERFILE, re.M)
    assert line, f"Dockerfile never copies {pkg.name}/pyproject.toml"
    assert f"{pkg.name}/uv.lock" in line.group(0), (
        f"Dockerfile copies {pkg.name}/pyproject.toml without its uv.lock; "
        f"`uv sync --frozen` would fail")


def test_source_is_copied_after_dependencies_are_installed():
    """The whole point of the layout: deps must install before source lands.

    If `COPY . /opt/bencher` moves back above the dependency sync, every source
    change rebuilds every virtualenv again.
    """
    deps_sync = next(
        line_number
        for line_number, run in _run_instruction_blocks()
        if _has_shell_command(run, DEPENDENCY_ONLY_SYNC)
    )
    full_copy = next(
        i for i, line in enumerate(DOCKERFILE.splitlines())
        if re.match(r"^COPY\s+\.\s+/opt/bencher\s*$", line)
    )
    assert deps_sync < full_copy, (
        "`COPY . /opt/bencher` must come after the dependency-only sync, "
        "otherwise a source edit invalidates the whole dependency build")


def test_dependency_sync_keeps_no_install_project_on_the_same_command():
    assert not _has_shell_command(
        "RUN uv sync && echo --no-install-project",
        DEPENDENCY_ONLY_SYNC,
    )


@pytest.mark.parametrize("command", [
    "RUN uv sync --frozen || echo --no-install-project",
    "RUN uv sync --frozen | echo --no-install-project",
])
def test_dependency_sync_keeps_no_install_project_before_pipe_boundaries(command):
    assert not _has_shell_command(command, DEPENDENCY_ONLY_SYNC)


def test_dependency_sync_does_not_count_inline_comment_text():
    assert not _has_shell_command(
        "RUN uv sync --frozen # --no-install-project",
        DEPENDENCY_ONLY_SYNC,
    )
    assert _has_shell_command(
        "RUN uv sync --find-links '#local' --no-install-project",
        DEPENDENCY_ONLY_SYNC,
    )


@pytest.mark.parametrize(("command", "pattern"), [
    (
        'RUN echo "setup; curl https://astral.sh/uv/${UV_VERSION}/install.sh | sh"',
        (
            r"curl\b[^;&|\n]*https://astral\.sh/uv/\$\{UV_VERSION\}/install\.sh"
            r"[^;&|\n]*\|[ \t]*sh\b"
        ),
    ),
    (
        "RUN echo 'setup && uv sync --no-install-project'",
        DEPENDENCY_ONLY_SYNC,
    ),
])
def test_shell_command_ignores_control_operators_inside_quoted_echo_text(
        command, pattern):
    assert not _has_shell_command(command, pattern)


def test_dockerignore_excludes_local_virtualenvs():
    """Excluding .venv keeps the context small AND stops the source copy from
    clobbering the virtualenvs built in the dependency layer."""
    assert re.search(r"^\*/\.venv/?$", DOCKERIGNORE, re.M), (
        ".dockerignore must exclude */.venv -- otherwise `COPY . /opt/bencher` "
        "overwrites the virtualenvs built earlier in the build")


@pytest.mark.parametrize("script", [
    "prefetch_libsvm.py", "prefetch_maxsat.py", "prefetch_svm.py", "prefetch_mopta.py",
])
def test_referenced_prefetch_scripts_exist(script):
    assert f"/opt/bencher-build/{script}" in DOCKERFILE, (
        f"docker/{script} exists but the Dockerfile does not run it")
    assert (REPO_ROOT / "docker" / script).is_file(), (
        f"Dockerfile runs {script} but docker/{script} does not exist")


PR_WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "pr-checks.yml").read_text()
DOCKER_BUILD_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "docker_build.yml").read_text()
SCAFFOLD_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "update_scaffold_version.yml").read_text()
UV_VENV_CHECK = (
    REPO_ROOT / ".github" / "scripts" / "verify_uv_venvs.sh").read_text()
LOCAL_SETUP_DOCS = {
    "README.md": (REPO_ROOT / "README.md").read_text(),
    "AGENTS.md": (REPO_ROOT / "AGENTS.md").read_text(),
    "CLAUDE.md": (REPO_ROOT / "CLAUDE.md").read_text(),
}


def _workflow_steps(workflow):
    """Return each top-level GitHub Actions step, including its nested fields."""
    return re.findall(
        r"(?ms)^      - .*?(?=^      - |\Z)",
        workflow,
    )


def _managed_python_env_is_set(step):
    return (
        'UV_MANAGED_PYTHON: "1"' in step
        and "UV_PYTHON_DOWNLOADS: never" in step
    )


def _step_run_command(step):
    """Return a step's shell command without mistaking quoted prose for code."""
    lines = step.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("        run:"):
            continue
        command = line.removeprefix("        run:").strip()
        if command != "|":
            return command
        return "\n".join(
            nested.removeprefix("          ")
            for nested in lines[index + 1:]
            if nested.startswith("          "))
    return ""


def _step_runs_uv_command(step, command):
    return bool(_has_shell_command("RUN " + _step_run_command(step), command))


def _setup_uv_versions(workflow):
    """Find the configured version for every setup-uv action in a workflow."""
    action_blocks = re.findall(
        r"(?ms)^\s*- (?:name: [^\n]+\n\s+)?uses: "
        r"astral-sh/setup-uv@[^\n]+.*?(?=^\s*- |\Z)",
        workflow,
    )
    return [
        re.search(r'(?m)^\s+version:\s*"([^"]+)"\s*$', block).group(1)
        for block in action_blocks
    ]


def _container_lane_packages():
    """Packages whose tests run inside the built image."""
    import tomllib
    out = []
    for pkg in PACKAGES:
        config = tomllib.loads((pkg / "pyproject.toml").read_text())
        tier = config.get("tool", {}).get("bencher", {}).get("ci", {}).get("tier")
        if tier == "container" and (pkg / "tests").is_dir():
            out.append(pkg)
    return out


def test_container_leg_mounts_the_shared_test_harness():
    """Container-tier tests import the harness from the repo-root tests/.

    That path resolves to /opt/bencher/tests inside the image, which
    .dockerignore deliberately keeps empty -- so the directory must be mounted
    or the tests fail at collection with ModuleNotFoundError. This exact
    omission broke the first container run.
    """
    assert '-v "$PWD/tests:/opt/bencher/tests:ro"' in PR_WORKFLOW, (
        "the container leg must mount the repo-root tests/ so package tests can "
        "import tests/grpc_harness.py")


def test_container_leg_mounts_each_package_test_directory():
    assert '-v "$PWD/$package/tests:/opt/bencher/$package/tests:ro"' in PR_WORKFLOW


@pytest.mark.parametrize("pkg", _container_lane_packages(),
                         ids=lambda p: p.name)
def test_container_lane_tests_only_reach_paths_the_image_has(pkg):
    """Their tests may reach outside the package only into tests/ -- the one
    directory the container leg mounts."""
    offenders = []
    for path in sorted((pkg / "tests").glob("*.py")):
        for line in path.read_text().splitlines():
            if "parents[2]" in line and '"tests"' not in line:
                offenders.append(f"{rel(path)}: {line.strip()}")
    assert not offenders, (
        "container-tier tests reach a repo path the image does not mount:\n"
        + "\n".join(offenders))


SDEF = (REPO_ROOT / "container.sdef").read_text()


def _entrypoint_command() -> str:
    """The command the image actually starts, from the Dockerfile ENTRYPOINT."""
    match = re.search(r'^ENTRYPOINT\s+\[(.*?)\]', DOCKERFILE, re.M)
    assert match, "Dockerfile has no ENTRYPOINT"
    return " ".join(part.strip().strip('"') for part in match.group(1).split(","))


def test_the_image_uses_the_bencher_server_venv_entrypoint():
    assert _entrypoint_command() == (
        "/opt/bencher/BencherServer/.venv/bin/python /entrypoint.py")


def test_sdef_startscript_matches_the_image_entrypoint():
    """Apptainer ignores Docker's ENTRYPOINT, so the sdef restates it by hand.

    If the two drift, the Docker image works and the Apptainer instance does not
    start -- a failure only the Apptainer tier would ever see.
    """
    command = _entrypoint_command()
    # Anchor to the section header at line start: the word also appears in the
    # file's prose comments, and matching those found an empty body.
    match = re.search(r'^%startscript\n(.*?)(?=^%|\Z)', SDEF, re.M | re.S)
    assert match, "container.sdef has no %startscript section"
    startscript = match.group(1)
    assert command in startscript, (
        f"container.sdef %startscript does not run the image's ENTRYPOINT "
        f"({command!r}); the instance would fail to start")


def test_readme_apptainer_template_starts_the_real_entrypoint():
    """The README's sdef template is a contract with users building their own.

    It documented `/docker-entrypoint.sh`, a file that does not exist in the
    image, so anyone copying it got a container that would not start.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    templates = re.findall(r'%startscript\s*\n\s*(.+)', readme)
    assert templates, "no %startscript in the README's Apptainer template"

    command = _entrypoint_command()
    wrong = [t for t in templates if command not in t]
    assert not wrong, (
        f"README's Apptainer template starts {wrong}, but the image's entrypoint "
        f"is {command!r}; a user copying it would get a container that never starts")


def test_the_image_uses_uv_managed_pythons():
    """Managed interpreters replace pyenv with uv's version-pinned installer."""
    assert 'ARG UV_VERSION="0.9.2"' in DOCKERFILE
    assert 'ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"' in DOCKERFILE
    run_blocks = _run_instruction_blocks()
    assert any(
        _has_shell_command(
            block,
            r"curl\b[^;&|\n]*https://astral\.sh/uv/\$\{UV_VERSION\}/install\.sh"
            r"[^;&|\n]*\|[ \t]*sh\b",
        )
        for _, block in run_blocks)
    assert any(
        (loop := re.search(
            r"\bfor[ \t]+version[ \t]+in\b.*?;[ \t]*do\b(.*?)\bdone\b",
            block,
            re.S,
        ))
        and _has_shell_command(
            loop.group(1), r'uv[ \t]+python[ \t]+install[ \t]+"\$version"')
        for _, block in run_blocks), (
        "Dockerfile must install each requested Python version with uv inside "
        "the version loop")
    assert "PYENV_" not in DOCKERFILE
    assert "pyenv" not in DOCKERFILE.lower()


def test_uv_installs_unique_pins_from_the_shared_cache():
    """The interpreter layer is pin-only and must stay warm across source edits."""
    assert re.search(
        r"for\s+version\s+in\s+\$\(sort -u "
        r"/opt/bencher/\*/\.python-version\)",
        DOCKERFILE,
    )
    assert "RUN --mount=type=cache,target=/root/.cache \\" in DOCKERFILE
    assert 'chmod -R a+rX "$UV_PYTHON_INSTALL_DIR"' in DOCKERFILE


def test_each_docker_sync_uses_the_matching_managed_interpreter():
    """Dependency and final project syncs cannot silently fetch another Python."""
    syncs = re.findall(
        r"UV_MANAGED_PYTHON=1\s+UV_PYTHON_DOWNLOADS=never"
        r"(?:\s*\\)?\s+"
        r'uv\s+sync\s+--python\s+"\$version"',
        DOCKERFILE,
    )
    assert len(syncs) == 2


@pytest.mark.parametrize("workflow", [
    PR_WORKFLOW,
    DOCKER_BUILD_WORKFLOW,
    SCAFFOLD_WORKFLOW,
])
def test_setup_uv_version_matches_the_image(workflow):
    """Docker and every CI path must use one audited uv release."""
    assert _setup_uv_versions(workflow) == ["0.9.2"] * len(
        _setup_uv_versions(workflow))
    assert _setup_uv_versions(workflow), "workflow does not install uv"


def test_e2e_python_install_reads_the_root_pin():
    """The e2e client must follow .python-version rather than a stale patch."""
    assert re.search(
        r'uv\s+python\s+install\s+"\$\(cat\s+\.python-version\)"',
        DOCKER_BUILD_WORKFLOW,
    )


@pytest.mark.parametrize("workflow", [PR_WORKFLOW, SCAFFOLD_WORKFLOW],
                         ids=["pr-checks", "update-scaffold"])
def test_lock_checking_workflows_install_every_pin(workflow):
    """`uv lock --check` needs each package's own pinned interpreter.

    Tier 0 and the scaffold bump both lock every package with downloads off,
    so installing only the root pin fails the 3.8 and 3.10 packages. The pins
    are read with awk, not cat: a pin file lacking a trailing newline would
    fuse with the next one into a bogus version like `3.11.133.8.20`.
    """
    assert re.search(
        r"uv\s+python\s+install\s+\$\(awk\s+'NF'\s+\.python-version"
        r"\s+\*/\.python-version\s*\|\s*sort -u\)",
        workflow,
    )


@pytest.mark.parametrize("workflow", [
    PR_WORKFLOW,
    DOCKER_BUILD_WORKFLOW,
    SCAFFOLD_WORKFLOW,
])
def test_ci_scopes_managed_python_to_sync_lock_and_run_steps(workflow):
    """Installs may download pins; later Python consumers may not."""
    python_steps = [
        step for step in _workflow_steps(workflow)
        if _step_runs_uv_command(step, r"uv\s+(?:sync|lock|run)\b")
    ]
    assert python_steps, "workflow has no uv sync, lock, or run steps"
    assert all(_managed_python_env_is_set(step) for step in python_steps)
    install_steps = [
        step for step in _workflow_steps(workflow)
        if _step_runs_uv_command(step, r"uv\s+python\s+install\b")
    ]
    assert install_steps, "workflow has no explicit uv Python install"
    assert not any(_managed_python_env_is_set(step) for step in install_steps)


@pytest.mark.parametrize("path, document", LOCAL_SETUP_DOCS.items())
def test_local_setup_docs_sync_with_the_selected_python_pin(path, document):
    """Local setup must select the pin it just installed, never discover one."""
    assert 'version="$(cat .python-version)"' in document, (
        f"{path} must derive a reusable version from .python-version")
    assert 'uv python install "$version"' in document, (
        f"{path} must install the selected version")
    syncs = [line for line in document.splitlines() if "uv sync" in line]
    assert syncs, f"{path} has no local uv sync command"
    assert all(
        'UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=never' in sync
        and '--python "$version"' in sync
        for sync in syncs
    ), f"{path} has a sync that does not explicitly select $version"


def test_ci_sync_and_lock_steps_select_the_installed_pin():
    """CI may download only in install steps, then must use the exact pin."""
    contract_sync = next(
        step for step in _workflow_steps(PR_WORKFLOW)
        if "Install contract-test dependencies" in step)
    assert re.search(
        r'version="\$\(cat \.python-version\)"\s*\n'
        r'\s*uv sync --python "\$version" --group dev',
        _step_run_command(contract_sync),
    )

    package_sync = next(
        step for step in _workflow_steps(PR_WORKFLOW)
        if "name: Install dependencies" in step)
    assert (
        'uv sync --python "${{ matrix.pkg.python }}" --locked --group dev'
        in _step_run_command(package_sync))

    e2e_sync = next(
        step for step in _workflow_steps(DOCKER_BUILD_WORKFLOW)
        if "Install the e2e client" in step)
    assert re.search(
        r'version="\$\(cat \.python-version\)"\s*\n'
        r'\s*uv sync --python "\$version" --group dev',
        _step_run_command(e2e_sync),
    )

    lock_step = next(
        step for step in _workflow_steps(SCAFFOLD_WORKFLOW)
        if "Update the pin and the lock in every package" in step)
    assert re.search(
        r'python_version="\$\(cat "\$pkg/\.python-version"\)"\s*\n'
        r'\s*\(\s*cd "\$pkg" && uv lock --python "\$python_version"'
        r' --refresh-package bencherscaffold \)',
        _step_run_command(lock_step),
    )

    scaffold_sync = next(
        step for step in _workflow_steps(SCAFFOLD_WORKFLOW)
        if "Run Tier 0 contract tests against the new scaffold" in step)
    assert re.search(
        r'version="\$\(cat \.python-version\)"\s*\n'
        r'\s*uv sync --python "\$version" --group dev',
        _step_run_command(scaffold_sync),
    )


def test_container_build_runs_pin_and_location_checks_before_each_e2e_suite():
    """Both runtimes must exercise the same uv-managed venv interpreters."""
    assert "for version_file in /opt/bencher/*/.python-version" in UV_VENV_CHECK
    assert "sys.version_info[:3]" in UV_VENV_CHECK
    assert "readlink -f" in UV_VENV_CHECK
    assert "/opt/uv-python/" in UV_VENV_CHECK

    docker_check = "docker exec -i bencher sh < .github/scripts/verify_uv_venvs.sh"
    sif_check = "apptainer exec bencher.sif sh < .github/scripts/verify_uv_venvs.sh"
    assert docker_check in DOCKER_BUILD_WORKFLOW
    assert sif_check in DOCKER_BUILD_WORKFLOW
    assert DOCKER_BUILD_WORKFLOW.index(docker_check) < DOCKER_BUILD_WORKFLOW.index(
        "Run e2e against the Docker container")
    assert DOCKER_BUILD_WORKFLOW.index(sif_check) < DOCKER_BUILD_WORKFLOW.index(
        "Run e2e against the Apptainer instance")


def test_docker_build_pr_paths_include_every_python_pin():
    """Changing any pin must trigger the packaging workflow."""
    match = re.search(
        r"(?m)^  pull_request:\n(?:    # [^\n]*\n)*"
        r"    paths:\n((?:      - [^\n]+\n)+)",
        DOCKER_BUILD_WORKFLOW,
    )
    assert match, "docker build workflow has no pull-request path filter"
    assert "'.python-version'" in match.group(1)
    assert "'*/.python-version'" in match.group(1)


@pytest.mark.parametrize("command", [
    "RUN echo setup # curl https://astral.sh/uv/${UV_VERSION}/install.sh | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh && echo fake | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh || echo fake | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh | echo fake | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh ; echo fake | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh & echo fake | sh",
    "RUN curl https://astral.sh/uv/${UV_VERSION}/install.sh\n echo fake | sh",
    "RUN echo setup || curl https://astral.sh/uv/${UV_VERSION}/install.sh | sh",
    "RUN echo setup | curl https://astral.sh/uv/${UV_VERSION}/install.sh | sh",
    "RUN echo setup & curl https://astral.sh/uv/${UV_VERSION}/install.sh | sh",
])
def test_versioned_uv_installer_requires_curl_to_pipe_directly_to_sh(command):
    installer = (
        r"curl\b[^;&|\n]*https://astral\.sh/uv/\$\{UV_VERSION\}/install\.sh"
        r"[^;&|\n]*\|[ \t]*sh\b"
    )
    assert not _has_shell_command(command, installer)


@pytest.mark.parametrize("body", [
    '# uv python install "$version"',
    'echo setup || uv python install "$version"',
    'echo setup | uv python install "$version"',
    'echo setup & uv python install "$version"',
])
def test_uv_python_install_loop_ignores_comments_and_shell_control_paths(body):
    block = f'RUN for version in 3.8 3.11; do\n  {body}\ndone'
    loop = re.search(
        r"\bfor[ \t]+version[ \t]+in\b.*?;[ \t]*do\b(.*?)\bdone\b",
        block,
        re.S,
    )
    assert loop
    assert not _has_shell_command(
        loop.group(1), r'uv[ \t]+python[ \t]+install[ \t]+"\$version"')


def test_apptainer_image_is_not_built_as_a_sandbox():
    """A --sandbox is a writable directory, so it reproduces none of the
    read-only-filesystem failures that justify testing Apptainer at all -- and
    the README tells users to build a .sif."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "docker_build.yml").read_text()
    # Match the command, not the word: the workflow explains in a comment why it
    # deliberately does not use --sandbox, and a bare substring check trips on it.
    assert not re.search(r'apptainer build[^\n]*--sandbox', workflow), (
        "the Apptainer tier must build a real .sif; a sandbox is writable and "
        "would hide every read-only failure this tier exists to catch")
