# ==================================================================================== #
# VARIABLES
# ==================================================================================== #

# Makefile Colors
PURPLE := \033[95m
BLUE := \033[94m
CYAN := \033[96m
GREEN := \033[92m
ORANGE := \033[93m
RED := \033[91m
ENDC := \033[0m
BOLD := \033[1m
UNDERLINE := \033[4m

# ==================================================================================== #
# MAKEFILE TARGETS
# ==================================================================================== #
.PHONY: install check test nox clean build serve-docs build-docs

install: ## install all group dependencies and install the pre-commit hooks
	@echo "$(PURPLE)--- Installing Environment ---$(ENDC)"
	@echo "$(BLUE) > Creating virtual environment and syncing dependencies...$(ENDC)"
	@uv sync --all-groups --all-extras
	@echo "$(BLUE) > Sorting dependencies with uv-sort...$(ENDC)"
	@uvx uv-sort
	@echo "$(BLUE) > Installing pre-commit hooks...$(ENDC)"
	@uvx pre-commit install
	@echo "$(GREEN)Install complete! Activate the venv with: source .venv/bin/activate$(ENDC)"

check: ## run code quality tools
	@echo "$(PURPLE)--- Running Code Quality Checks ---$(ENDC)"
	@echo "$(BLUE) > Checking lock file consistency...$(ENDC)"
	@uv lock --locked
	@echo "$(BLUE) > Running pre-commit checks...$(ENDC)"
	@uv run pre-commit run -a
	@echo "$(GREEN)All checks passed!$(ENDC)"

test: ## test the code with pytest
	@echo "$(PURPLE)--- Running Tests ---$(ENDC)"
	@echo "$(BLUE) > Running pytest with coverage report...$(ENDC)"
	@uv run pytest --cov --cov-config=pyproject.toml --cov-report=html --color=yes
	@echo "$(GREEN)Tests finished!$(ENDC)"

nox: ## run nox session
	@echo "$(PURPLE)--- Running Nox ---$(ENDC)"
	@echo "$(BLUE) > Running noxfile...$(ENDC)"
	@uv run nox
	@echo "$(GREEN)All nox checks finishe!$(ENDC)"

clean: ## remove build artifacts and caches
	rm -rf .nox .pytest_cache .ruff_cache __pycache__ dist site htmlcov
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

serve-docs: ## serve docs locally at localhost:8001
	@echo "$(PURPLE)--- Serving Documentation ---$(ENDC)"
	@uv run --group docs mkdocs serve -a localhost:8001

build-docs: ## build static docs site to site/
	@echo "$(PURPLE)--- Building Documentation ---$(ENDC)"
	@uv run --group docs mkdocs build
	@echo "$(GREEN)Docs built to site/$(ENDC)"

build: ## build wheel to dist/
	@echo "$(PURPLE)--- Building Wheel ---$(ENDC)"
	@uv build --wheel --out-dir dist/
	@echo "$(GREEN)Wheel built to dist/$(ENDC)"
