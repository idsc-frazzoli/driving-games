
run-dg-experiments: build
	mkdir -p $(out-docker)
	docker run -it --user $$(id -u) \
		-v $(PWD)/$(out-docker):/out $(tag) \
		dg-demo -o /out/dgfact --reset -c "rparmake" \
		--games "simple_int_2p_sets","simple_int_3p_sets","multilane_int_4p_sets","multilane_int_5p_sets" \
		--solvers "solver-2-pure-security_mNE-naive-noextra","solver-2-pure-security_mNE-fact1-noextra","solver-2-pure-security_mNE-fact2-noextra"

#,"multilane_int_4p_sets","multilane_int_5p_sets" \
#		--games "4way_int_2p_sets" \  #,"4way_int_3p_sets","multilane_int_4p_sets","multilane_int_5p_sets" \
 #		--solvers "solver-2-pure-security_mNE-fact1-noextra"  #,"solver-2-pure-security_mNE-naive-noextra","solver-2-pure-security_mNE-fact2-noextra"
#

run-posets-exp: build
	mkdir -p $(out-docker)
	docker run -it \
		-v $(PWD)/scenarios:/driving_games/scenarios:ro \
		-v $(PWD)/$(out-docker):/out $(tag) \
		posets-exp -o /out/posets --reset -c "rparmake"



run-crashing_experiments: build
	mkdir -p $(out-docker)
	docker run -it --user $$(id -u) \
		-v $(PWD)/scenarios:/driving_games/scenarios:ro \
		-v $(PWD)/$(out-docker):/out $(tag) \
		crash-exp -o /out/crash --reset -c "rparmake"
