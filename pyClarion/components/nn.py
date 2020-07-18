"""Tools for constructing and managing bottom level neural nets."""


__all__ = ["Map2VectorEncoder"]


class Map2VectorEncoder(object):
    """
    Converts between activation maps and vectors for neural network I/O.

    Consistently encodes activation maps as vectors and vectors as activation 
    maps.
    """

    def __init__(self, encoding, default_strength=0.0):
        """
        Initialize a microfeature to vector converter.

        :param encoding: Expected nodes in ordered iterable.
        :param default_strength: The default activation strength.
        """

        self.encoding = encoding
        self.default_strength = default_strength

    def encode(self, activation_map):
        """
        Encode node activation map into activation vector.

        :param vector: An activation vector.
        """
        vector = []
        for node in self.encoding:
            vector.append(activation_map.get(node, self.default_strength))
        return vector

    def decode(self, vector):
        """
        Decode activation vector into node activation map.

        :param vector: An activation vector.
        """

        output = {}
        for index, activation in enumerate(vector):
            output[self.encoding[index]] = activation
        return output
