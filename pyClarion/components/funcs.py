from ..numdicts.numdicts import Unary, Ternary, Aggregator

cam = Aggregator(lambda *xs: max(0.0, *xs) + min(0.0, *xs), 0.0)
least_squares_cost = Ternary(lambda est, tgt, msk: msk * (est - tgt) ** 2)
