This folder constains the source code for the semester project 'Motion planning in urban driving scenarios via evaluation of homotopy classes'.
In folder MIQP is the Mixed Integer Quadratic Programming forlumation and testing codes for solving the joint optimization of players in an intersection scenatio by evaluating plans within each candidate homotopy classes.

Structure of the codes:
MIQP
    scenario.py creates either a simple intersection scenario by defining number of the players and their current states, or loading a commonroad map and placing the players in it.

    Folder forces_def contains codes that define a MIQP model for FORCES Pro and generate the solver. It also provides the utility functions to solve the problem in a receding horizon fashion, visualize plots and generate reports. The functions are called by forces_def_test/solver_test.py to test the optimization for a given homotopy class.

    homotopy/homotopy.py contains the defination of a homotopy class(main part is the heuristic function) and the function to enumerate all homotopy classes given a scenario, compute their corresponding heiristics and rank them.
    homotopy/report.py generates reports for evaluating all homotopy classes.

    homotpy_test/homotopy_test.py solves all homotopy classes in the sequence suggested by the heuristics and generate test reports.
    homotopy_test/no_homo_test.py additionally generates FORCES Pro solver that solves a similar MIQP problem that finds the global optimal solution for collision avoidance(without pre-defining any homotopy class.). This can be used as a baseline to evaluate the computation efficiency of the proposed method.

    Folder utils contains necessary functions to predict the reference trajectory from the current states of a player, compute the intersection points between the predicted trajectories of the players and visualize the debugging plots. The functions are tested in utils.utils_test.py.

Run a test:
test solving MIQP given a homotopy class choice:
    execute homotopies/MIQP/forces_def_test/solver_test.py
    (modify variable h for different homotopy class choices)

test evaluating all homotopy classes and solving MIQP for each of them:
    execute homotopies/MIQP/homotpy_test/homotopy_test.py

test solving MIQP globally without defining any homotopy class
    execute homotopies/MIQP/homotpy_test/no_homo_test.py
