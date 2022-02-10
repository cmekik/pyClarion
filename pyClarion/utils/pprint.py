"""A copy of stdlib pprint augmented to support pyClarion objects."""


__all__ = ["PrettyPrinter", "pprint", "pformat"]


from ..numdicts import NumDict

from typing import ClassVar
import pprint as _pprint


class PrettyPrinter(_pprint.PrettyPrinter):

    _dispatch: ClassVar[dict] = _pprint.PrettyPrinter._dispatch # type: ignore
    
    def _pprint_numdict(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        name = type(object).__name__
        indent += len(name) + 1
        end = [
            " " * indent, 
            'c=', _pprint.saferepr(object.c), 
            ')']
        
        stream.write(name + '(')
        self._pprint_dict(object, stream, indent, allowance, context, level) # type: ignore
        stream.write(',\n')
        stream.write("".join(end))

    _dispatch[NumDict.__repr__] = _pprint_numdict

def pprint(object, stream=None, indent=1, width=80, depth=None, *,
           compact=False):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""

    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth, compact=compact
    )
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, *, compact=False):
    """Format a Python object into a pretty-printed representation."""
    
    printer = PrettyPrinter(
        indent=indent, width=width, depth=depth, compact=compact
    )

    return printer.pformat(object)
