"""Defines state for 4-way crossing"""

import pomdp_py


# States
# Ego State has one component less since route is known
class EgoState(pomdp_py.ObjectState):
    def __init__(self, speed, progress):
        super().__init__("Ego", {'speed': speed, 'progress': progress})

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def speed(self):
        return self.attributes['speed']

    @property
    def progress(self):
        return self.attributes['progress']


# Other agents have route in their state
class ExtendedState(pomdp_py.ObjectState):
    def __init__(self, pname, speed, progress, route):
        super().__init__(str(pname), {'speed': speed, 'progress': progress, 'route': route})

    @property
    def speed(self):
        return self.attributes['speed']

    @property
    def progress(self):
        return self.attributes['progress']

    @property
    def route(self):
        return self.attributes['route']


# Observations
class EgoObservation(pomdp_py.Observation):
    def __init__(self, speed, progress):
        super().__init__("Ego", {'speed': speed, 'progress': progress})

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class OtherObservation(pomdp_py.Observation):
    def __init__(self, pname, speed, x_coord, y_coord):
        # why not following line?
        #super().__init__(str(pname), {'speed': speed, 'x_coord': x_coord, 'y_coord': y_coord})
        self.pname = pname
        self.speed = speed
        self.x_coord = x_coord
        self.y_coord = y_coord

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


# Actions
class MotionAction(pomdp_py.Action):
    # allowed actions
    DEC1 = -2.0
    DEC2 = -1.0
    CONTINUE = 0.0
    ACC1 = 1.0

    def __init__(self, acc):

        if acc not in {MotionAction.DEC1, MotionAction.DEC2, MotionAction.CONTINUE, MotionAction.ACC1}:
            raise ValueError("Invalid acceleration %s" % str(acc))

        self.acc = acc

        super().__init__("Accelerate %s" % acc) # what is this exactly needed for?


    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
