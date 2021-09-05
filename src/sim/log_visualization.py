import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from sim import SimLog

colorsIdx = {'P0': 'rgb(0,215,215)', 'P1': 'rgb(0,215,100)',
             'P2': 'rgb(0,0,215)', 'P3': 'rgb(215,0,0)',
             'P4': 'rgb(0,215,0)', 'P5': 'rgb(215,215,0)',
             'P6': 'rgb(215,0,215)', 'P7': 'rgb(100,215,215)',
             'P8': 'rgb(100,0,215)', 'P9': 'rgb(0,100,215)',
             'P10': 'rgb(100,215,100)', 'P11': 'rgb(100,100,215)',
             'P12': 'rgb(215,100,100)', 'P13': 'rgb(200,100,200)',
             'P14': 'rgb(200,100,0)', 'P15': 'rgb(0,150,0)',
             'P16': 'rgb(0,215,50)', 'P17': 'rgb(150,215,50)',
             'P18': 'rgb(215,70,100)', 'P19': 'rgb(215,50,150)',
             'P20': 'rgb(50,70,130)', 'P21': 'rgb(215,70,130)',
             'P22': 'rgb(0,140,215)', 'P23': 'rgb(0,215,215)',
             'P24': 'rgb(150,150,20)', 'P25': 'rgb(215,72,184)',
             'P26': 'rgb(160,215,0)', 'P27': 'rgb(215,100,0)',
             'P28': 'rgb(90,180,215)', 'P29': 'rgb(215,26,98)',
             'Ego': 'rgb(90,215,180)', 'EGO': 'rgb(215,90,180)',
             }


def get_input_plots(log: SimLog) -> Figure:

    n_inputs = []
    agent_ID = []

    # Number of inputs for each agent
    for player in log:
        n_inputs.append(len(log[player].actions.values[00].idx))
        agent_ID.append(type(log[player].actions.values[00]))

    # unique_agent_ID = set(agent_ID)
    # unique_agent_count = len(unique_agent_ID)
    fig = make_subplots(
        rows=int(np.ceil(max(n_inputs) / 2)), cols=2, column_widths=[0.5, 0.5])

    for player in log:
        x_label = log[player].actions.timestamps
        actions = log[player].actions.values[00].idx

        for inputs in actions:
            commands = log[player].actions.values
            if inputs == 'acc':
                y_label = [commands[item].acc for item in range(len(commands))]
            elif inputs == 'ddelta':
                y_label = [commands[item].ddelta for item in range(len(commands))]
            elif inputs == 'dtheta':
                y_label = [commands[item].dtheta for item in range(len(commands))]
            else:
                y_label = []

            row = int(np.floor(actions[inputs] / 2)) + 1
            col = np.mod(actions[inputs], 2) + 1
            fig.add_trace(
                go.Scatter(
                    x=x_label,
                    y=y_label,
                    line=dict(width=1, dash="dot", color=colorsIdx[player]),
                    mode="lines+markers",
                    name=player,
                ),
                row=row,
                col=col
            )
            fig.update_yaxes(row=row, col=col)
            fig.update_xaxes(title_text=inputs, row=row, col=col)

    fig.update_layout(title_text="Inputs")

    return fig


def get_state_plots(log: SimLog) -> Figure:

    n_states = []

    # Number of inputs for each agent
    for player in log:
        n_states.append(len(log[player].states.values[00].idx))

    fig = make_subplots(
        rows=int(np.ceil(max(n_states) / 2)), cols=2, column_widths=[0.5, 0.5])

    for player in log:
        x_label = log[player].states.timestamps
        states_vec = log[player].states.values[00].idx

        for sx in states_vec:
            states = log[player].states.values
            if sx == 'x':
                y_label = [states[item].x for item in range(len(states))]
            elif sx == 'y':
                y_label = [states[item].y for item in range(len(states))]
            elif sx == 'theta':
                y_label = [states[item].theta for item in range(len(states))]
            elif sx == 'vx':
                y_label = [states[item].vx for item in range(len(states))]
            elif sx == 'vy':
                y_label = [states[item].vy for item in range(len(states))]
            elif sx == 'dtheta':
                y_label = [states[item].dtheta for item in range(len(states))]
            elif sx == 'delta':
                y_label = [states[item].delta for item in range(len(states))]
            else:
                y_label = []

            row = int(np.floor(states_vec[sx] / 2)) + 1
            col = np.mod(states_vec[sx], 2) + 1
            fig.add_trace(
                go.Scatter(
                    x=x_label,
                    y=y_label,
                    line=dict(width=1, dash="dot", color=colorsIdx[player]),
                    mode="lines+markers",
                    name=player,
                ),
                row=row,
                col=col
            )
            fig.update_yaxes(row=row, col=col)
            fig.update_xaxes(title_text=sx, row=row, col=col)

    fig.update_layout(title_text="States")

    return fig
