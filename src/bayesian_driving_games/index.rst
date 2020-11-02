
Bayesian Driving Games package
=========================================

.. automodule:: bayesian_driving_games
   :members:
   :undoc-members:

Structures
----------

.. automodule:: bayesian_driving_games.structures
   :members:

.. automodule:: bayesian_driving_games.structures_solution
   :members:

bayesian_games.structures:
Includes classes BayesianGamePlayer and BayesianVehicleState

bayesian_games.structures_solution:
Includes class BayesianGameNode


Game generation
---------------

.. automodule:: bayesian_driving_games.create_joint_game_tree
   :members:


Game rewards
---------------

.. automodule:: bayesian_driving_games.bayesian_driving_rewards
   :members:


Game solution
---------------

.. automodule:: bayesian_driving_games.solution
   :members:

.. automodule:: bayesian_driving_games.sequential_rationality
   :members:

.. automodule:: bayesian_driving_games.preprocess
   :members:



BAYESIAN GAMES:

Game:
- Normal Game, except two things:
- New Payoff structure: for every type combination a payoff:
    Mapping[Set[PlayerType], Mapping[PlayerName, BirdCosts]]
- New BayesianGamePlayer:
    - inherits GamePlayer
    - types_of_others: List[PlayerType]
    - types_of_myself: List[PlayerType]
    - prior: Poss[PlayerType] (prior over types_of_others)

Game graph generation:
- BayesianGameNode:
    - Inherits GameNode
    - game_node_belief:
        -   for every player, a belief about the every type of the other player:
            Mapping[PlayerName, Poss[PlayerType]]
        - initialized at the prior. Prior: ProbPoss[PlayerTypes]

solving algorithm:
- in solution.py:
    game_solution = solve_game_bayesian2(
        game=gp.game,
        gg=gg,
        solver_params=gp.solver_params,
        jss=initials,
    )
- solve_game_bayesian2 consists of a loop with three steps:
    1.) solve the game given the current beliefs
    2.) compare resulting strategy to old strategy
    3.) if strategies in 2.) different: Assign new beliefs according to new strategy (TODO)

1.) - uses _solve_bayesian_game(sc: SolvingContext, js0: JointState), a recursive
      function with output SolvedGameNode[X, U, Y, RP, RJ, SR].
    - Every SolvedGameNode has a:
        - states = currentJointState,
        - solved = where we can go from here,
        - va = a ValueAndAction (see below),
        - ur = Used resources (at the moment not implemented)

    - the ValueAndAction has normally a "game_value: Dict[PlayerName, UncertainCombined]",
        we need however a game_value: Dict[Set[PlayerName, PlayerType], UncertainCombined]"
        due to the different payoffs.

    - The best strategy is solved in solve_sequential_rationality(sc, gn, solved: Dict[JointPureActions, game_value):
        - Important: We do not need a strategy for each player, but a strategy for a player and his type:
            => Make new PlayerNames that include the PlayerType
        - Solve like done in the example in the report

    - End up with a strategy for a Player-Type pair for every node and a expected game_value


2.) simple test

3.) todo


ToDo:
- belief assignment
- off-the-path beliefs
- we will have personal payoffs in driving games
- see if mixed strategies work
- Driving games bayesian Payoffs
- What if we have multiple equilibria (adjust structure)




