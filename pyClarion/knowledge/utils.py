__all__ = ["get_dyads", "get_lhs", "get_rhs"]


from typing import overload

from .compounds import Chunks, Rules, AnyDVPairs


@overload
def get_dyads(sort: Chunks) -> AnyDVPairs:
    pass

@overload
def get_dyads(sort: Rules) -> tuple[AnyDVPairs, AnyDVPairs]:
    pass

def get_dyads(sort):
    match sort:
        case Chunks():
            return sort._dyads_
        case Rules():
            return sort._lhs_._dyads_, sort._rhs_._dyads_
        case _:
            TypeError(f"Unexpected object of type {type(sort)}")


def get_chunks(rules: Rules) -> tuple[Chunks, Chunks]:
    return rules._lhs_, rules._rhs_


def get_lhs(rules: Rules) -> Chunks:
    return rules._lhs_


def get_rhs(rules: Rules) -> Chunks:
    return rules._rhs_