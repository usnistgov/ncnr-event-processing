import numpy as np

from .candor import cleanup as candor_cleanup
from .vsans import cleanup as vsans_cleanup
from .sans import cleanup as sans_cleanup

CLEANUP_FNS = {
    'candor': candor_cleanup,
    'vsans': vsans_cleanup,
    'sans': sans_cleanup,
    'ngb30msans': sans_cleanup,
    '10msans': sans_cleanup,
}