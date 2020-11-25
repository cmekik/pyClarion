"""A copy of stdlib pprint augmented to support pyClarion objects."""


__all__ = ["PrettyPrinter", "pprint", "pformat"]


from ..base.numdicts import BaseNumDict
from ..components import Chunks, Rules, BLAs

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
        end = [" " * indent, 'default=', _pprint.saferepr(object.default),')']
        
        stream.write(name + '(')
        self._pprint_dict(object, stream, indent, allowance, context, level)
        stream.write(',\n')
        stream.write("".join(end))

    _dispatch[BaseNumDict.__repr__] = _pprint_numdict

    def _pprint_Chunks(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        name = type(object).__name__
        indent += len(name) + 1

        stream.write(name + '(')
        self._pprint_dict(object, stream, indent, allowance, context, level)
        stream.write(')')

    _dispatch[Chunks.__repr__] = _pprint_Chunks

    def _pprint_Chunk(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        
        name = type(object).__name__
        features = object.features
        weights = object.weights

        indent += len(name) + 1
        findent = indent + len("features=")
        windent = indent + len("weights=")

        stream.write(name + '(')
        stream.write("features=")
        self._dispatch[type(features).__repr__](
            self, features, stream, findent, allowance, context, level
        )
        stream.write(",\n" + " " * indent + "weights=")
        self._dispatch[type(weights).__repr__](
            self, weights, stream, windent, allowance, context, level
        )
        stream.write(')')

    _dispatch[Chunks.Chunk.__repr__] = _pprint_Chunk

    def _pprint_Rules(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        name = type(object).__name__
        indent += len(name) + 1
        
        stream.write(name + '(')
        self._pprint_dict(object, stream, indent, allowance, context, level)
        stream.write(')')

    _dispatch[Rules.__repr__] = _pprint_Rules

    def _pprint_Rule(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        
        name = type(object).__name__
        conc = object.conc
        weights = object.weights

        indent += len(name) + 1
        windent = indent + len("weights=")

        stream.write('<' + name + ' ')
        stream.write("conc="+_pprint.saferepr(conc))
        stream.write(",\n" + " " * indent + "weights=")
        self._dispatch[type(weights).__repr__](
            self, weights, stream, windent, allowance, context, level
        )
        stream.write('>')

    _dispatch[Rules.Rule.__repr__] = _pprint_Rule

    def _pprint_BLAs(
        self, object, stream, indent, allowance, context, level
    ):

        write = stream.write
        name = type(object).__name__
        indent += len(name) + 1
        
        stream.write(name + '(')
        self._pprint_dict(object._dict, stream, indent, allowance, context, level)
        stream.write(')')

    _dispatch[BLAs.__repr__] = _pprint_BLAs


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
