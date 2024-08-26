from typing import TypeAlias
import numpy as np

Signal: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]

def rms(x: Signal) -> float:
    return np.sqrt(np.sum(x**2)/len(x))


def db(x: float) -> float:
    return 20 * np.log10(x)


def rmsdb(x: Signal) -> float:
    return db(rms(x))
