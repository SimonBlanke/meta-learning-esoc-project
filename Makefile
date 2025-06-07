build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y meta-learn-esoc-project
	rm -fr build dist *.egg-info

install-test-requirements:
	python -m pip install .[test]

install-build-requirements:
	python -m pip install .[build]

install-all-extras:
	python -m pip install .[all_extras]

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

test-pytest:
	python -m pytest --durations=10 -x -p  no:warnings tests/; \