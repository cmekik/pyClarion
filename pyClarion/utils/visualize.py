from __future__ import annotations
from typing import Sequence
import warnings

from ..base import Structure, uris
from . import inspect

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("Install matplotlib to enable visualization tools.")
else:
    def adjacency_matrix(
        ax: plt.Axes, s: Structure, exclude: Sequence[str] = ()
    ) -> plt.Axes:
        ms = sorted([m.path for m in s.modules() if m.path not in exclude])
        ms_rev = list(reversed(ms))
        links = sorted(set((tgt, src.partition(uris.FSEP)[0]) 
            for tgt, src in inspect.links(s)))
        lbls = [uris.relativize(path, s.path) for path in ms]
        coords = [(ms.index(src), ms_rev.index(tgt)) 
            for tgt, src in links if tgt not in exclude and src not in exclude]
        x, y = list(zip(*coords))
        ax.set_aspect("equal")
        ax.set(
            xlabel="Inputs",
            xticks=list(range(len(lbls))),
            yticks=list(range(len(lbls))),
            yticklabels=reversed(lbls),)
        ax.set_xlabel('Sources')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xticklabels(lbls, rotation=90)
        ax.set_ylabel('Targets', rotation=90)    
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.scatter(x, y, c="black")
        ax.set_axisbelow(True)
        ax.grid()
        if len(exclude) > 0:
            exc = sorted(
                f"\'{uris.relativize(path, s.path)}\'" for path in exclude)
            ax.annotate(f"Excluded: {', '.join(exc)}", 
                (0, - mpl.rcParams["font.size"] - 2), xycoords="axes points")
        return ax


