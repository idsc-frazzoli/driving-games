from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from games import PlayerName
from sim import SimulationLog
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleCommands


def create_vectors_log(log: SimulationLog, players: List[PlayerName]):

    states_log = dict((el, []) for el in VehicleStateDyn.idx)
    actions_log = dict((el, []) for el in VehicleCommands.idx)
    states_log['t'] = []
    actions_log['t'] = []
    players_states_log = dict((player, states_log) for player in players)
    players_actions_log = dict((player, actions_log) for player in players)

    for player in players:
        for timestamps in log:
            players_states_log[player]['x'].append(log.at(timestamps)[player].state.x)
            players_states_log[player]['y'].append(log.at(timestamps)[player].state.y)
            players_states_log[player]['theta'].append(log.at(timestamps)[player].state.theta)
            players_states_log[player]['vx'].append(log.at(timestamps)[player].state.vx)
            players_states_log[player]['vy'].append(log.at(timestamps)[player].state.vy)
            players_states_log[player]['dtheta'].append(log.at(timestamps)[player].state.dtheta)
            players_states_log[player]['delta'].append(log.at(timestamps)[player].state.delta)
            players_states_log[player]['t'].append(timestamps)
            if timestamps > 0:
                players_actions_log[player]['acc'].append(log.at(timestamps)[player].actions.acc)
                players_actions_log[player]['ddelta'].append(log.at(timestamps)[player].actions.ddelta)
                players_actions_log[player]['t'].append(timestamps)

    return players_states_log, players_actions_log


def get_input_plots(players_actions_log) -> Figure:

    n_players = len(players_actions_log)
    n_inputs = dict((player, len(players_actions_log[player]) - 1) for player in players_actions_log)
    names = dict((player, list(players_actions_log[player].keys())) for player in players_actions_log)

    fig = make_subplots(
        rows=int(np.ceil(2 / 2) * n_players), cols=2, column_widths=[0.5, 0.5])
    for player in players_actions_log:
        x_label = players_actions_log[player]['t']
        names[player].remove('t')
        for i in range(n_inputs[player]):
            row = int(np.floor(i / 2)) + 1
            col = np.mod(i, 2) + 1
            fig.add_trace(
                go.Scatter(
                    x=x_label,
                    y=players_actions_log[player][names[player][i]],
                    line=dict(width=1, dash="dot"),
                    mode="lines+markers",
                    name=names[player][i],
                ),
                row=row,
                col=col
            )
            fig.update_yaxes(row=row, col=col)

    fig.update_layout(title_text="Inputs")

    return fig


def get_state_plots(players_states_log) -> Figure:

    n_players = len(players_states_log)
    n_states = dict((player, len(players_states_log[player]) - 1) for player in players_states_log)
    names = dict((player, list(players_states_log[player].keys())) for player in players_states_log)

    fig = make_subplots(
        rows=int(np.ceil(7 / 2) * n_players), cols=2, column_widths=[0.5, 0.5])
    for player in players_states_log:
        x_label = players_states_log[player]['t']
        names[player].remove('t')
        for i in range(n_states[player]):
            row = int(np.floor(i / 2)) + 1
            col = np.mod(i, 2) + 1
            fig.add_trace(
                go.Scatter(
                    x=x_label,
                    y=players_states_log[player][names[player][i]],
                    line=dict(width=1, dash="dot"),
                    mode="lines+markers",
                    name=names[player][i],
                ),
                row=row,
                col=col
            )
            fig.update_yaxes(row=row, col=col)

    fig.update_layout(title_text="States")

    return fig