import typing

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def get_mode(classes_μm: np.ndarray,
             classes_φ: np.ndarray,
             frequency: np.ndarray,
             is_geometric: bool = False) -> float:
    """
    Get the mode size of the grain size distribution.

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.
    is_geometric: `bool` (default `False`)
        If `True` the mode size is in μm, else it's in φ.

    ## Returns

    mode: `float`
        The mode size of a grain size distribution.

    """
    max_pos = np.unravel_index(np.argmax(frequency), frequency.shape)
    if is_geometric:
        return classes_μm[max_pos]
    else:
        return classes_φ[max_pos]

def get_modes(classes_μm: np.ndarray,
              classes_φ: np.ndarray,
              frequency: np.ndarray,
              is_geometric=False, trace=0.01) \
        -> typing.Tuple[typing.Tuple[float], typing.Tuple[float]]:
    """
    Get the mode sizes (peaks) of a grain size distribution. The number may be greater than 1.

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.
    is_geometric: `bool` (default `False`)
        If `True` the mode size is in μm, else it's in φ.
    trace: `float` (default `0.01`, `1%`)
        If the peak's frequency is less than `trace`, it will be ignored.

    ## Returns

    modes: `tuple`
        The mode sizes of a grain size distribution.
    frequencies: `tuple`
        The frequencies of corresponding mode sizes.

    """
    peaks, _ = find_peaks(frequency)
    position = classes_μm[peaks] if is_geometric else classes_φ[peaks]
    values = frequency[peaks]
    not_trace = np.greater_equal(values, trace)
    return tuple(position[not_trace]), tuple(values[not_trace])

def get_cumulative_frequency(frequency: np.ndarray, expand=False):
    """
    Transform the frequency distribution to cumulative frequency.

    ## Parameters

    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.
    expand: `bool` (default `False`)
        If `True`, `0.0` and `1.0` will be added at the head and tail, respectively.

    ## Returns

    culmulative_frequency: `np.ndarray`
        A numpy array contains the culmulative frequencies.

    """
    cumulative_frequency = []
    cumulative = 0.0
    if expand:
        cumulative_frequency.append(0.0)
    for i, f in enumerate(frequency):
        cumulative += f
        cumulative_frequency.append(cumulative)
    if expand:
        cumulative_frequency.append(1.0)
    return np.array(cumulative_frequency)

def get_reverse_phi_ppf(classes_φ: np.ndarray, frequency: np.ndarray) -> interp1d:
    """
    Get the reversed percent point function (PPF) of a grain size distribution.
    Because the grain size classes in φ is isometric, it's more suitable for interpolation.
    And because coarser particles have smaller φ values, the PPF is reversed.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    ppf: `scipy.interpolate.interp1d`
        A `interp1d` object which is callable and could be regarded as a PPF function.

    """

    interval = np.mean(classes_φ[1:] - classes_φ[:-1])
    expand_classes = np.linspace(classes_φ[0] - interval, classes_φ[-1]+interval, len(classes_φ)+2)
    cumulative_frequency = get_cumulative_frequency(frequency, expand=True)
    cumulative_frequency = np.array(cumulative_frequency)
    ppf = interp1d(cumulative_frequency, expand_classes, kind="slinear")
    return ppf

def get_interval_φ(classes_φ: np.ndarray)-> float:
    """
    Get the interval of grain size classes.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.

    ## Returns

    interval_φ: `float`
        The interval in φ.

    """

    return abs((classes_φ[0]-classes_φ[-1]) / (len(classes_φ)-1))

def convert_μm_to_φ(size_μm: typing.Union[float, np.ndarray]):
    """
    Convert the grain size from μm space to φ space.

    ## Parameters

    size_μm: `float` or `np.ndarray`
        The grain size in μm.

    ## Returns

    size_φ: `np.ndarray`
        The grain size in φ.

    """

    return -np.log2(size_μm / 1000.0)

def convert_φ_to_μm(size_φ: np.ndarray):
    """
    Convert the grain size from φ space to μm space.

    ## Parameters

    size_φ: `float` or `np.ndarray`
        The grain size in φ.

    ## Returns

    size_μm: `np.ndarray`
        The grain size in μm.

    """

    return np.exp2(-size_φ) * 1000.0
