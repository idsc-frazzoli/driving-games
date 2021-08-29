CIRCLE_NODE_INDEX ?= 0
CIRCLE_NODE_TOTAL ?= 1

out=out
out-docker=out-docker
coverage_dir=$(out)/coverage
tr=$(out)/test-results
xunit_output=$(tr)/nose-$(CIRCLE_NODE_INDEX)-xunit.xml

tag=driving_games

test_packages=dg_commons_tests,driving_games_tests,preferences_tests,games_tests,sim_tests,possibilities_tests,crash_tests,trajectory_games_tests
cover_packages=$(test_packages),dg_commons,driving_games,preferences,games,games_zoo,possibilities,sim,crash,trajectory_games

parallel=--processes=8 --process-timeout=1000 --process-restartworker
coverage=--cover-html --cover-html-dir=$(coverage_dir) --cover-tests --with-coverage --cover-package=$(cover_packages)

xunitmp=--with-xunitmp --xunitmp-file=$(xunit_output)
extra=--rednose --immediate


all:
	@echo "You can try:"
	@echo
	@echo "  make build run"
	@echo "  make docs "
	@echo "  make test coverage-combine coverage-report"



clean:
	coverage erase
	rm -rf $(out) $(coverage_dir) $(tr)

test: clean
	mkdir -p  $(tr)
	DISABLE_CONTRACTS=1 nosetests $(extra) $(coverage) src  -v --nologcapture $(xunitmp)

test-docker: clean
	mkdir -p  $(tr)
	docker run -it \
		-v ${PWD}/src:/driving_games/src:ro \
		-v $(PWD)/$(out-docker):/driving-games/$(tr) $(tag) \
		nosetests $(extra) $(coverage)  src  -v --nologcapture $(xunitmp)

test-parallel: clean
	mkdir -p  $(tr)
	DISABLE_CONTRACTS=1 nosetests $(extra) $(coverage) src  -v --nologcapture $(parallel)


test-parallel-circle:
	DISABLE_CONTRACTS=1 \
	NODE_TOTAL=$(CIRCLE_NODE_TOTAL) \
	NODE_INDEX=$(CIRCLE_NODE_INDEX) \
	nosetests $(coverage) $(xunitmp) src  -v  $(parallel)

coverage-combine:
	coverage combine

build:
	docker build -t $(tag) .

build-no-cache:
	docker build --no-cache -t $(tag) .

run:
	mkdir -p $(out-docker)
	docker run -it --user $$(id -u) \
		-v $(PWD)/$(out-docker):/out $(tag) \
		dg-demo -o /out/result --reset -c "rparmake"

run-with-mounted-src:
	mkdir -p $(out-docker)
	docker run -it --user $$(id -u) \
		-v $(PWD)/src:/driving_games/src:ro \
		-v $(PWD)/$(out-docker):/out $(tag) \
		dg-demo -o /out/result --reset -c "rparmake"

black:
	black -l 120 --target-version py38 src

coverage-report:
	coverage html  -d $(coverage_dir)


###### Docs
docs:
	sphinx-build src $(out)/docs

docs-docker: build
	mkdir -p $(out-docker)/docs
	docker run -it --rm --user $$(id -u)\
		-v ${PWD}/src:/driving_games/src:ro \
		-v ${PWD}/$(out-docker)/docs:/driving-games/$(out)/docs $(tag) \
		sphinx-build src /driving-games/$(out)/docs


include makefiles/Makefile.version