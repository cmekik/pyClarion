import doctest

import pyClarion.base.packets
import pyClarion.base.links

doctest.testmod(pyClarion.base.packets, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.links, optionflags=doctest.ELLIPSIS)