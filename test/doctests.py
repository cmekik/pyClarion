import doctest

import pyClarion.base.node
import pyClarion.base.packet
import pyClarion.base.channel
import pyClarion.base.junction
import pyClarion.base.selector
import pyClarion.base.effector
import pyClarion.base.connector

doctest.testmod(pyClarion.base.node, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.packet, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.channel, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.junction, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.selector, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.effector, optionflags=doctest.ELLIPSIS)
doctest.testmod(pyClarion.base.connector, optionflags=doctest.ELLIPSIS)