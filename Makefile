PYTHON ?= python3
.PHONY: help setup native doctor test quick smoke verify-best-known verify-examples
help:
	@printf '%s\n' 'make setup           create .venv and install dependencies (network)' 'make native          compile dsatur_native.so' 'make doctor          local dependency/backend check' 'make test            all tests' 'make quick           fast tests' 'make smoke           bounded search+verification' 'make verify-best-known verify stored configuration certificates'
setup:
	bash scripts/setup.sh
native:
	$(PYTHON) scripts/build_native.py
doctor:
	$(PYTHON) scripts/doctor.py
test:
	$(PYTHON) scripts/run_tests.py --suite all
quick:
	$(PYTHON) scripts/run_tests.py --suite quick
smoke:
	$(PYTHON) scripts/smoke.py --backend native
verify-examples: verify-best-known
verify-best-known:
	$(PYTHON) scripts/verify_examples.py --backend python
