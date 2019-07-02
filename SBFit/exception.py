import warnings


class Error(Exception):
    """Basic custom exception class"""

    def __init__(self, message):
        self.message = message


class MismatchingError(Error):
    pass


class InvalidNumberError(Error):
    pass


class InvalidValueError(Error):
    pass
