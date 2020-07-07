import numpy as np


""" Step 1: Observe => If observation=true => normal game (under ML-assumption), else Step 2

Step 2: Solve all 4 games => We need a payoff at the end

Step 3: Calculate probabilities of all 4 games

Step 4: Weigh the payoffs, choose right action. """

if __name__ == '__main__':

    a1, a2 = 12, 2
    b1, b2 = 17, 12
    side_left = 8
    side_right = 14

    t1_left = (side_left-a1)/(b1-a1)
    t1_right = (side_right-a1)/(b1-a1)

    if 0 < t1_left < 1:
        res_left = a2 + t1_left*(b2-a2)
    else:
        res_left = 12
    if 0 < t1_right < 1:
        res_right = a2 + t1_right * (b2 - a2)
    else:
        res_right = 12


    if res_left < side_left:
        print("not seen down left")
    elif res_left > side_right:
        print("not seen up left")
    elif res_right < side_left:
        print("not seen down right")
    elif res_right > side_right:
        print("not seen up right")
    else:
        print ("seen")

