import doctest

import pyClarion.base.knowledge
import pyClarion.base.packet
import pyClarion.base.connector

doctest.testmod(pyClarion.base.knowledge, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.packet, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.connector, optionflags=doctest.ELLIPSIS)