# Enhanced Makefile for managing the Conda environment tda-nn-separability

# Parameters
ENV_NAME := tda-nn-separability
PYTHON_VERSION := 3.10
ENV_FILE := environment.yaml

# Default goal
.PHONY: all
all: init install export

# Check if the conda command is available
CONDA := $(shell command -v conda 2> /dev/null)

# Initialize the Conda environment
.PHONY: init
init:
ifndef CONDA
	$(error "conda is not available, please install Miniconda or Anaconda.")
endif
	@echo "Creating the Conda environment if it doesn't exist..."
	@conda env list | grep -q '^$(ENV_NAME) ' || \
	conda create --yes --prefix $(ENV_NAME) python=$(PYTHON_VERSION)

# Install packages from an environment file or manually specified
.PHONY: install
install: init
	@echo "Installing necessary packages into the Conda environment..."
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "Using $(ENV_FILE) to install packages..."; \
		conda env update --name $(ENV_NAME) --file $(ENV_FILE) --prune; \
	else \
		echo "No $(ENV_FILE) found. Installing default packages..."; \
		conda run --name $(ENV_NAME) conda install --yes numpy pandas scipy matplotlib; \
	fi

# Export the current environment to a YAML file, excluding build-specific fields
.PHONY: export
export:
	@echo "Exporting the Conda environment to $(ENV_FILE)..."
	@conda env export --name $(ENV_NAME) --no-builds | grep -v "^prefix: " > $(ENV_FILE)

# Update all packages in the Conda environment
.PHONY: update
update: init
	@echo "Updating all packages in the Conda environment..."
	@conda run --name $(ENV_NAME) conda update --all --yes

# Run tests using pytest within the Conda environment
.PHONY: test
test: init
	@echo "Running tests..."
	pytest tests -v

# Clean up __pycache__ directories and *.pyc files
.PHONY: clean
clean:
	@echo "Cleaning up __pycache__ directories and *.pyc files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + > /dev/null 2>&1
	@find . -type f -name "*.pyc" -delete > /dev/null 2>&1

