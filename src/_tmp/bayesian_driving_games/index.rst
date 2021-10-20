
Bayesian Driving Games package
=========================================

.. automodule:: bayesian_driving_games
   :members:
   :undoc-members:


The solution algorithm works as follows: A game is defined with all parameters like in a normal driving game. Different
to a normal driving game is that one has to define types for both players. The reward functions (personal and joint)
have to be in such a way that all type combinations get a reward. I wrote two personal and one joint function so that
one can see how it could be done (see below "Game Rewards"). The game is then used in "Game Generation" to build a
game tree and populate it with beliefs and rewards. The game tree is then solved using the solving algorithm
described in my (Michael's) thesis (see "Game Solution" below). First a solving step consisting of backwards solving
(from the leaves of the tree upwards) with expected values (weighted with the beliefs). Important to note is that
every type of every player has to choose an action, and the combination of all these actions together result in a
value. All action combinations are than compared for a Nash equilibrium of a stage in the game. After solving the
game tree, the solution strategy is used to update the beliefs. Then the tree is solved again with new beliefs. This
loop is repeated until the strategy does not change anymore.



Structures
----------
Three new classes: BayesianVehicleState, which is a normal VehicleState plus a type of a player. BayesianGameNode is
a normal GameNode plus a belief. PlayerType is a class for the type of a player.

.. automodule:: bayesian_driving_games.structures
   :members:

.. automodule:: bayesian_driving_games.structures_solution
   :members:


Game generation
---------------

.. automodule:: bayesian_driving_games.create_joint_game_tree
   :members:

.. automodule:: bayesian_driving_games.preprocess
   :members:


Game rewards
---------------
The rewards have to be in such a way that every type combination gets a payoff.

.. automodule:: bayesian_driving_games.bayesian_driving_rewards
   :members:


Game solution
---------------
Solution algorithm is explained at the beginning of this documentation and in the thesis. The different functions are
described here below.

.. automodule:: bayesian_driving_games.solution
   :members:

.. automodule:: bayesian_driving_games.sequential_rationality
   :members:
