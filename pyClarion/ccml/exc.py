class CCMLException(Exception):
    pass


class CCMLSyntaxError(CCMLException):
    pass


class CCMLNameError(CCMLException):
    pass


class CCMLTypeError(CCMLException):
    pass


class CCMLValueError(CCMLException):
    pass