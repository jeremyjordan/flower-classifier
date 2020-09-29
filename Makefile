init:
	pip install --upgrade -r requirements-dev.txt
	pip install -e .
	pre-commit install

format:
	black . --line-length=120
	isort . --multi-line VERTICAL_HANGING_INDENT --trailing-comma --line-width=120
	flake8

test:
	python3 -m pytest tests/ -m "not download" --cov=flower_classifier/ --cov-report term-missing --cov-config=tests/.coveragerc

check:
	pre-commit run --all-files

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
