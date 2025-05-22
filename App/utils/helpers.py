"""Utility helper functions for the application."""

from typing import Tuple


def seconds_to_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def generate_track_color(track_id: int) -> Tuple[float, float, float, float]:
    """Generate a unique color for a track ID for Kivy (RGBA format).
    
    Parameters
    ----------
    track_id : int
        Track ID to generate color for
        
    Returns
    -------
    Tuple[float, float, float, float]
        RGBA color tuple with values in range 0-1
    """
    hue = (track_id * 0.618033988749895) % 1.0
    
    h = hue
    s = 0.8
    v = 0.9
    
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m, 1.0) 