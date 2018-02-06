ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	if [ ! -d venv ]; then virtualenv -q venv; fi
	venv/bin/python -m pip install --upgrade -r requirements.txt
	echo "\nexport PYTHONPATH=\$$PYTHONPATH:$(ROOT_DIR)" >> venv/bin/activate
