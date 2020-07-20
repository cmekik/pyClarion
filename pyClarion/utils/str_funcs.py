import collections.abc as abc
from pyClarion.base.symbols import Symbol


__all__ = ["pstr_iterable", "pstr_iterable_cb"]


def pstr_iterable(iterable, cb, cbargs=None, indent=4, level=0):
    """
    Return a pretty string representation of iterable.

    Places each item on a separate line with appropriate indentation for ease of 
    reading. Intended for logging and reporting of iterable contents.

    Deals recursively with nested iterables, if iterables are of type:
        list, dict, collections.abc.Set, collections.abc.Mapping

    For mappings, recursion only occurs on values, not on keys.

    :param iterable: iterable to be delimited.
    :param cb: A callback for converting non-iterable contents of c into 
        strings. 
    :param indent: Number indentation spaces.
    """

    recursion_targets = (tuple, list, abc.Set, abc.Mapping)

    if not isinstance(iterable, recursion_targets):
        raise ValueError("{!r} is not supported.".format(type(iterable)))

    outer = " " * indent * level
    r_outer = outer
    inner = outer + " " * indent

    if cbargs is None:
        cbargs = {}

    if isinstance(iterable, (abc.Set, abc.Mapping)):
        ldelim, rdelim = "{", "}"
    elif isinstance(iterable, list):
        ldelim, rdelim = "[", "]"
    else:
        ldelim, rdelim = "(", ")"

    if len(iterable) > 0:
        item_strs = []
        if isinstance(iterable, abc.Mapping):
            for k, v in iterable.items():
                k_str = cb(k, **cbargs)
                if isinstance(v, recursion_targets):
                        v_str = pstr_iterable(
                            iterable=v, 
                            cb=cb, 
                            cbargs=cbargs,
                            indent=indent, 
                            level=level + 1,
                        )
                else:
                    v_str = cb(v, **cbargs)
                item_strs.append("{k}: {v}".format(k=k_str, v=v_str))
        
        elif isinstance(iterable, (list, tuple, abc.Set)):
            for i in iterable:
                if isinstance(i, recursion_targets):
                    i_str = pstr_iterable(
                        iterable=i, 
                        cb=cb, 
                        cbargs=cbargs,
                        indent=indent, 
                        level=level + 1, 
                    )
                else:
                    i_str = cb(i, **cbargs)
                item_strs.append(i_str)


        content = ",\n".join([inner + i_str for i_str in item_strs])
        cstr = (
            "{ldelim}\n{content}\n{r_outer}{rdelim}"
        ).format(
            ldelim=ldelim, 
            rdelim=rdelim, 
            content=content, 
            r_outer=r_outer
        )        

    else:
        cstr = "{ldelim}{rdelim}".format(
            ldelim=ldelim, 
            rdelim=rdelim
        )

    return cstr


def pstr_iterable_cb(obj, digits=None):

    if isinstance(obj, Symbol):
        return str(obj)
    else:
        s = obj
        if digits is not None:
            try:
                s = round(obj, digits)
            except TypeError:
                pass

        return repr(s)