

include Makefile.version


test_packages=zcities
cover_packages=zcities,zcities_tests

parallel=--processes=8 --process-timeout=1000 --process-restartworker
coverage=--cover-html --cover-tests --with-coverage --cover-package=$(cover_packages)

xunitmp=--with-xunitmp --xunitmp-file=test-results/nose-$(CIRCLE_NODE_INDEX)-xunit.xml
extra=--rednose --immediate

tr=test-results
coverage_dir=out/coverage

all:
	echo


clean:
	rm -f .coverage
	rm -rf cover
	rm -rf $(tr)
	mkdir -p  $(tr)
	rm -rf $(out) $(coverage_dir) .coverage .coverage.*

test:
	$(MAKE) clean
	nosetests $(extra) $(coverage) $(xunitmp) src  -v


test-parallel:
	$(MAKE) clean
	nosetests $(extra) $(coverage) src  -v  $(parallel)

test-parallel-failed:
	$(MAKE) clean
	nosetests  $(extra)  $(coverage) src  -v  $(parallel)

test-parallel-circle:
	NODE_TOTAL=$(CIRCLE_NODE_TOTAL) NODE_INDEX=$(CIRCLE_NODE_INDEX) nosetests $(coverage) $(xunitmp) src  -v  $
	(parallel)

test-failed:
	$(MAKE) clean
	nosetests $(extra)  --with-id --failed $(coverage) src  -v


tag=driving_games

build:
	docker build -t $(tag) .
run:
	mkdir -p out-docker
	docker run -it -v $(PWD)/src:/driving_games/src:ro -v $(PWD)/out-docker:/out $(tag) \
		dg-demo -o /out/result --reset -c "rparmake"
