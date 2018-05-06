TARGETS = core python27 python36

.PHONY: ${TARGETS}

build: ${TARGETS}

core:
	make --directory core

python27: core
	make --directory python27

python36: core
	make --directory python36
