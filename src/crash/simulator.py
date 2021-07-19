from crash.simulator_structures import SimContext, LogEntry, SimObservations


class Simulator:
    last_observations: SimObservations

    def pre_update(self, sim_context: SimContext):
        self.last_observations.time = sim_context.time
        self.last_observations.players = {}
        for player_name, model in sim_context.models.items():
            self.last_observations.players.update({player_name: model.state})

    def update(self, sim_context: SimContext):
        sim_context.log[sim_context.time] = {}
        for player_name, model in sim_context.models.items():
            actions = sim_context.players[player_name].get_commands(self.last_observations)
            new_state = model.update(actions, dt=sim_context.param.dt)
            sim_context.log[sim_context.time].update({player_name: LogEntry(state=new_state, actions=actions)})
        # todo check if sim context gets updates properly or it needs to be returned

    def post_update(self, sim_context: SimContext):

        # todo here we check for collisions and termination (end of sim time and so on)

        sim_context.time += sim_context.param.dt
        if sim_context.time > 5: # todo temporary we just simulate 5 seconds
            sim_context.sim_terminated = True

        ...
