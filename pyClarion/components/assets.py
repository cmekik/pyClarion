from pyClarion.base.realizers import Assets
from pyClarion.components.datastructures import Chunks, Rules


class BasicAgentAssets(Assets):

    def __init__(self, chunks: Chunks = None):

        self.chunks = chunks if chunks is not None else Chunks()


class NACSAssets(Assets):

    def __init__(self, rules: Rules = None):

        self.rules = rules if rules is not None else Rules()
