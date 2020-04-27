
CIRCLE_NODE_INDEX ?= 0
CIRCLE_NODE_TOTAL ?= 1
include Makefile.version


out=out
coverage_dir=$(out)/coverage
tr=$(out)/test-results


test_packages=driving_games_tests,preferences_tests,games_tests
cover_packages=driving_games,preferences,games

parallel=--processes=8 --process-timeout=1000 --process-restartworker
coverage=--cover-html --cover-tests --with-coverage --cover-package=$(cover_packages)

xunitmp=--with-xunitmp --xunitmp-file=$(tr)/nose-$(CIRCLE_NODE_INDEX)-xunit.xml
extra=--rednose --immediate


all:
	echo


clean:
	coverage erase 
	rm -f .coverage
	rm -rf cover
	rm -rf $(tr)
	rm -rf $(out) $(coverage_dir) 

test:
	$(MAKE) clean
	mkdir -p  $(tr)
	nosetests $(extra) $(coverage) $(xunitmp) src  -v
	coverage combine

test-parallel:
	$(MAKE) clean
	mkdir -p  $(tr)
	nosetests $(extra) $(coverage) src  -v  $(parallel)
	coverage combine

test-parallel-circle:
	NODE_TOTAL=$(CIRCLE_NODE_TOTAL) NODE_INDEX=$(CIRCLE_NODE_INDEX) nosetests $(coverage) $(xunitmp) src  -v  $
	(parallel)
	coverage combine


# test-parallel-failed:
# 	$(MAKE) clean
# 	nosetests  $(extra)  $(coverage) src  -v  $(parallel)

# test-failed:
# 	$(MAKE) clean
# 	nosetests $(extra)  --with-id --failed $(coverage) src  -v


tag=driving_games

build:
	docker build -t $(tag) .

build-no-cache:
	docker build --no-cache -t $(tag) .


run:
	mkdir -p out-docker
	docker run -it -v $(PWD)/out-docker:/out $(tag) \
		dg-demo -o /out/result --reset -c "rparmake"

run-with-mounted-src:
	mkdir -p out-docker
	docker run -it -v $(PWD)/src:/driving_games/src:ro -v $(PWD)/out-docker:/out $(tag) \
		dg-demo -o /out/result --reset -c "rparmake"

black:
	black -l 100 --target-version py37 src

coverage-report: 
	coverage html  -d $(coverage_dir)
