 
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

Run using:

    $ dg-demo --reset
     
   


 
