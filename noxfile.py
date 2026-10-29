import nox

nox.options.sessions = ("ruff", "pyrefly", "bandit")

nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv|virtualenv"

SILENT_DEFAULT = True
SILENT_CODE_MODIFIERS = False

PACKAGE_LOCATION = "."
CODE_LOCATIONS = PACKAGE_LOCATION
PYTHON_VERSIONS = ["3.12", "3.13"]
PYPY3_VERSION = "pypy3"
LATEST_PYTHON = PYTHON_VERSIONS[-1]


@nox.session(python=PYTHON_VERSIONS, tags=["lint", "format"])
def ruff(session: nox.Session) -> None:
    """Lint and format with ruff."""
    args = session.posargs or (PACKAGE_LOCATION,)
    _install(session, "ruff")
    _run(session, "ruff", "check", *args)
    _run_code_modifier(session, "ruff", "format", *args)


@nox.session(python=LATEST_PYTHON, tags=["typecheck"])
def pyrefly(session: nox.Session) -> None:
    """Verify types using pyrefly."""
    args = session.posargs or ("aef_bng",)
    session.install(".[spark]", "pyrefly")
    _run(session, "pyrefly", "check", *args)


@nox.session(python="3.12", tags=["test"])
def tests(session: nox.Session) -> None:
    """Run the unit suite against the locked dev + spark environment, as CI does."""
    # dev is a dependency group, not an extra, so `install(".[dev]")` can't provide it.
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--group",
        "dev",
        "--extra",
        "spark",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        external=True,
    )
    # Not _run: external=True would fall back to whichever pytest is on PATH.
    session.run("pytest", "-m", "unit", *session.posargs)


@nox.session(python=PYTHON_VERSIONS, tags=["security"])
def bandit(session: nox.Session) -> None:
    """Scan for common security issues with bandit."""
    args = session.posargs or ("aef_bng",)
    _install(session, "bandit")
    _run(session, "bandit", *args)


def _install(session: nox.Session, *args: str) -> None:
    if args:
        session.install(*args)


def _run(
    session: nox.Session,
    target: str,
    *args: str,
    silent: bool = SILENT_DEFAULT,
) -> None:
    session.run(target, *args, external=True, silent=silent)


def _run_code_modifier(session: nox.Session, target: str, *args: str) -> None:
    _run(session, target, *args, silent=SILENT_CODE_MODIFIERS)
