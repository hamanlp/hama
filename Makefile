BUN ?= bun
PYTHON ?= python3
ZIG ?= zig
UV ?= uv
PY_SRC_DIR := $(CURDIR)/python
PY_UV_VENV := $(PY_SRC_DIR)/.venv
PY_UV_PYTHON := $(PY_UV_VENV)/bin/python

.PHONY: install install-js install-python build build-js build-zig test test-js test-python test-zig clean

install: install-js install-python

install-js:
	cd typescript && $(BUN) install

install-python:
	@if command -v $(UV) >/dev/null 2>&1; then \
		echo "Using uv to manage Python dependencies"; \
		$(UV) python install 3.12 >/dev/null; \
		$(UV) venv $(PY_UV_VENV) --python 3.12 >/dev/null; \
		cd $(PY_SRC_DIR) && $(UV) pip install --python $(PY_UV_PYTHON) -e ".[dev]"; \
	else \
		echo "uv not found; falling back to $(PYTHON)"; \
		cd $(PY_SRC_DIR) && $(PYTHON) -m pip install -e ".[dev]"; \
	fi

build: build-zig build-js

build-js:
	cd typescript && $(BUN) run build

build-zig:
	$(ZIG) build

test: test-zig test-js test-python

test-zig:
	$(ZIG) build test

test-js:
	cd typescript && $(BUN) test

test-python:
	@if [ -x $(PY_UV_PYTHON) ]; then \
		cd $(PY_SRC_DIR) && $(PY_UV_PYTHON) -m pytest; \
	else \
		cd $(PY_SRC_DIR) && $(PYTHON) -m pytest; \
	fi

clean:
	rm -rf typescript/dist python/build python/dist python/.pytest_cache python/.venv zig-cache zig-out
