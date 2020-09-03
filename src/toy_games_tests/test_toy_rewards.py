from toy_games.toy_rewards import BirdJointReward
from toy_games_tests.toy_games_tests_zoo import game2


def test_bird_joint_reward():
    bjr = BirdJointReward(max_stages=2, subgames=game2.subgames, row_player="p1", col_player="p2")
    print(bjr.mat_payoffs)
    # todo finish off to check proper distribution of payoffs
