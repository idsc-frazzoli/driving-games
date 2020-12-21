from reprep import Report

from .static_game import StaticGame


def report_game_visualization(game: StaticGame) -> Report:
    viz = game.game_vis
    r = Report("vis")
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    viz.plot_player(player_name=player_name, state=state)
                viz.plot_actions(player=player)

    return r
