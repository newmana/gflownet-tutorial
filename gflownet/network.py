import matplotlib.cm as cm
import matplotlib.pyplot as matplotlib_pyplot
import torch

from .face import Face

class Network:

    @staticmethod
    def plot(unique_states, transitions, model=None):
        lens = [len([i for i in unique_states if len(i) == j]) for j in range(4)]
        levels = [sorted([i for i in unique_states if len(i) == j]) for j in range(4)]
        fig = matplotlib_pyplot.figure(figsize=(8, 8))
        face2pos = {}
        for i, (level, L) in enumerate(zip(levels, lens)):
            for j, face in enumerate(level):
                fig.add_axes([j / L, i / 4, 1 / L, 1 / 6])
                f = Face(face)
                f.draw_face()
                face2pos[f.face_hash()] = (j / L + 0.5 / L, i / 4)
        ax = fig.add_axes([0, 0, 1, 1])
        matplotlib_pyplot.sca(ax)
        matplotlib_pyplot.gca().set_facecolor((0, 0, 0, 0))
        matplotlib_pyplot.xlim(0, 1)
        matplotlib_pyplot.ylim(0, 1)
        for a, b in transitions[1:]:
            if not len(b):
                continue
            pa, pb = face2pos[a.face_hash()], face2pos[b.face_hash()]
            la = int(pa[1] * 4)
            lb = int(pb[1] * 4)
            ws = [1 / 6, 1 / 6, 0.13, 0.11]

            if model is not None:
                flow_state = model(torch.tensor(a.face_hash()).float())
                flow_action = flow_state[Face.sorted_keys.index([i for i in b if i not in a][0])].item()
                c = cm.brg(flow_action / 3)
            else:
                c = None

            matplotlib_pyplot.arrow(pa[0], pa[1] + ws[la], pb[0] - pa[0], pb[1] - pa[1] - ws[lb], head_width=0.01,
                                    width=0.003,
                                    ec=c, fc=c,
                                    length_includes_head=True)
            matplotlib_pyplot.axis('off')

        if model is not None:
            ax = fig.add_axes([1, 0.2, 0.05, 0.6])
            matplotlib_pyplot.sca(ax)
            fig.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=3), cmap=cm.brg), cax=ax,
                       label='Edge Flow');
