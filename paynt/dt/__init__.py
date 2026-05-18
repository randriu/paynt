__version__ = "unknown"

try:
    from .._version import __version__
except ImportError:
    # We're running in a tree that doesn't have a _version.py, so we don't know what our version is.
    pass

def version():
    return __version__

from .api import *
from .task import DtTask
from .result import DtResult
from .factory import DtColoredMdpFactory
from .synthesizer import DtSynthesizer
from . import dtnest