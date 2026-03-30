from enum import Enum


class Color(Enum):
    red = 1
    green = 2
    blue = 3


class ContextType(Enum):
    DISABLED = 1
    ALL = 2
    SELECTED = 3
    PREVIOUS = 4

