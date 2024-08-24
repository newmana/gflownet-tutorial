import matplotlib.pyplot as matplotlib_pyplot
import numpy as numpy

class Face:
    patches = {
      'smile': lambda: matplotlib_pyplot.gca().add_patch(matplotlib_pyplot.Polygon(numpy.stack([numpy.linspace(0.2,0.8), 0.3-numpy.sin(numpy.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
      'frown': lambda: matplotlib_pyplot.gca().add_patch(matplotlib_pyplot.Polygon(numpy.stack([numpy.linspace(0.2,0.8), 0.15+numpy.sin(numpy.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
      'left_eb_down': lambda: matplotlib_pyplot.gca().add_line(matplotlib_pyplot.Line2D([0.15, 0.35], [0.75,0.7], color=(0,0,0))),
      'right_eb_down': lambda: matplotlib_pyplot.gca().add_line(matplotlib_pyplot.Line2D([0.65, 0.85], [0.7,0.75], color=(0,0,0))),
      'left_eb_up': lambda: matplotlib_pyplot.gca().add_line(matplotlib_pyplot.Line2D([0.15, 0.35], [0.7,0.75], color=(0,0,0))),
      'right_eb_up': lambda: matplotlib_pyplot.gca().add_line(matplotlib_pyplot.Line2D([0.65, 0.85], [0.75,0.7], color=(0,0,0))),
    }
    sorted_keys = sorted(patches.keys())

    def __init__(self, patches):
        self.patches = patches

    def draw_face(self):
      Face.base_face()
      for patch in self.patches:
        Face.patches[patch]()
      matplotlib_pyplot.axis('scaled')
      matplotlib_pyplot.axis('off')

    def has_overlap(self):
        # Can't have two overlapping eyebrows!
        if 'left_eb_down' in self.patches and 'left_eb_up' in self.patches:
            return True
        if 'right_eb_down' in self.patches and 'right_eb_up' in self.patches:
            return True
        # Can't have two overlapping mouths!
        if 'smile' in self.patches and 'frown' in self.patches:
            return True
        return False

    def face_reward(self):
        if self.has_overlap():
            return 0
        eyebrows = 'left_eb_down', 'left_eb_up', 'right_eb_down', 'right_eb_up'
        # Must have exactly two eyebrows
        if sum([i in self.patches for i in eyebrows]) != 2:
            return 0
        # We want twice as many happy faces as sad faces so here we give a reward of 2 for smiles
        if 'smile' in self.patches:
            return 2
        if 'frown' in self.patches:
            return 1  # and a reward of 1 for frowns
        # If we reach this point, there's no mouth
        return 0

    def __eq__(self, other):
        return isinstance(other, Face) and set(self.patches) == set(other.patches)

    def __hash__(self):
        return hash(tuple(sorted(self.patches)))

    def __iter__(self):
        return iter(self.patches)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]

    def __lt__(self, other):
        if not isinstance(other, Face):
            return NotImplemented
        return tuple(sorted(self.patches)) < tuple(sorted(other.patches))

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return not (self > other)

    def __ge__(self, other):
        return not (self < other)

    @staticmethod
    def base_face():
        matplotlib_pyplot.gca().add_patch(matplotlib_pyplot.Circle((0.5, 0.5), 0.5, fc=(.9, .9, 0)))
        matplotlib_pyplot.gca().add_patch(matplotlib_pyplot.Circle((0.25, 0.6), 0.1, fc=(0, 0, 0)))
        matplotlib_pyplot.gca().add_patch(matplotlib_pyplot.Circle((0.75, 0.6), 0.1, fc=(0, 0, 0)))

    @staticmethod
    def enumerate_states_transitions(keys):
        enumerated_states = []
        transitions = []
        def recursively_enumerate(s):
            if s.has_overlap():
                return
            for i in keys:
                if i not in s.patches:
                    recursively_enumerate(Face(s.patches + [i]))
            enumerated_states.append(s)
            transitions.append((s.patches[:-1], s.patches))
        recursively_enumerate(Face([]))
        return enumerated_states, transitions
