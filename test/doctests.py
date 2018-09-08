import doctest

import pyClarion.base.node
import pyClarion.base.packet
import pyClarion.base.channel
import pyClarion.base.junction
import pyClarion.base.selector
import pyClarion.base.effector
import pyClarion.base.connector

doctest.testmod(pyClarion.base.node)
doctest.testmod(pyClarion.base.packet)
doctest.testmod(pyClarion.base.channel)
doctest.testmod(pyClarion.base.junction)
doctest.testmod(pyClarion.base.selector)
doctest.testmod(pyClarion.base.effector)
doctest.testmod(pyClarion.base.connector)