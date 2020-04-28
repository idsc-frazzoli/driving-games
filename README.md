
# Code quality

[![CircleCI](https://circleci.com/gh/idsc-frazzoli/driving-games.svg?style=svg&circle-token=8bb1a7723db3a72ed58a7c2aa93ee088b43b1e80)](https://circleci.com/gh/idsc-frazzoli/driving-games) See [details here](https://circleci.com/gh/idsc-frazzoli/driving-games).

<!-- Note: there is a "branch" in the url -->

[![codecov](https://codecov.io/gh/idsc-frazzoli/driving-games/branch/master/graph/badge.svg?token=w8Sk4CKFpI)](https://codecov.io/gh/idsc-frazzoli/driving-games) (for master) - See [details](https://codecov.io/gh/idsc-frazzoli/driving-games).

(Need to be logged in with Github account in both cases) 

# Driving Games tutorial

Reachable states for player 1:

<img src="pics/player1.png" style="width: 80%; margin-left: auto; margin-right: auto">

Jointly reachable states for two players. 
Red is the initial state.
Magenta are collision states. Green are states where two players are active.
Yellow and blue are states in which there is only one agent. 
Black are terminating states for the single-player game.

<img src="pics/game.png" style="width: 80%; margin-left: auto; margin-right: auto">


## Docker installation and run

With Docker:

    $ make build
    
    $ make run

The results are going to be written in the `out-docker` dir.

## Without Docker

Requires Python >= 3.7.
    
Run manually the instructions in the Dockerfile.

Run tests using:

	$ make clean test coverage-combine coverage-report

Run some games using:

    $ dg-demo --reset
     
   


 
