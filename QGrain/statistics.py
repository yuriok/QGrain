__all__ = ["mode", "modes", "to_cumulative", "reversed_phi_ppf", "interval_phi", "to_phi", "to_microns", "cm",
           "arithmetic", "geometric", "logarithmic", "logarithmic_fw57", "geometric_fw57", "scale_description",
           "proportions_gsm", "proportions_ssc", "proportions_bgssc", "all_proportions",
           "group_gsm_folk54", "group_ssc_folk54", "group_folk54", "GROUP_BP12_SYMBOL_MAP",
           "group_gsm_bp12", "group_ssc_bp12", "group_bp12", "major_statistics", "all_statistics"]

import string
from typing import *

import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def mode(classes: ndarray,
         classes_phi: ndarray,
         distribution: ndarray,
         is_geometric: bool = False) -> float:
    """
    Get the mode size of a grain size distribution.

    :param classes: The grain size classes in microns.
    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :param is_geometric: If it‘s `True`, the returned mode size is in microns, else it's in phi values.
    :return: The mode size of a grain size distribution.
    """
    max_pos = np.unravel_index(np.argmax(distribution), distribution.shape)
    if is_geometric:
        return classes[max_pos]
    else:
        return classes_phi[max_pos]


def modes(classes: ndarray,
          classes_phi: ndarray,
          distribution: ndarray,
          is_geometric=False, trace=0.01) -> \
        Tuple[Sequence[float], Sequence[float]]:
    """
    Get the modes of a grain size distribution. **The number of modes may be greater than 1.**

    :param classes: The grain size classes in microns.
    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :param is_geometric: If it's `True`, the returned mode size is in microns, else it's in phi values.
    :param trace: If the peak frequency is less than `trace`, it will be ignored.
    :returns:
        modes: The mode sizes of a grain size distribution.
        frequencies: The frequencies of corresponding mode sizes.
    """
    peaks, _ = find_peaks(distribution)
    position = classes[peaks] if is_geometric else classes_phi[peaks]
    values = distribution[peaks]
    not_trace = np.greater_equal(values, trace)
    return tuple(position[not_trace]), tuple(values[not_trace])


def to_cumulative(distribution: ndarray, expand=False) -> ndarray:
    """
    Get the cumulative distribution.

    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :param expand: If `True`, `0.0` and `1.0` will be added at the head and tail, respectively.
    :return: A numpy array contains the cumulative frequencies.
    """
    if expand:
        cumulative = np.zeros(len(distribution) + 2)
        cumulative[0] = 0.0
        cumulative[-1] = 1.0
        cumulative[1:-1] = np.cumsum(distribution)
    else:
        cumulative = np.cumsum(distribution)
    return cumulative


def reversed_phi_ppf(classes_phi: ndarray, distribution: ndarray) -> \
        Callable[[Union[int, float, ndarray]], Union[int, float, ndarray]]:
    """
    Get the reversed percent point function (PPF) of a grain size distribution.
    Because the grain size classes in phi values is isometric, it's more suitable for interpolation.
    And because coarser particles have smaller phi values, the PPF is reversed.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: A scipy `interp1d` object which is callable and can be regarded as a PPF function.
    """
    interval = interval_phi(classes_phi)
    expand_classes = np.linspace(classes_phi[0] - interval, classes_phi[-1] + interval, len(classes_phi) + 2)
    cumulative = to_cumulative(distribution, expand=True)
    cumulative = np.array(cumulative)
    ppf = interp1d(cumulative, expand_classes, kind="slinear")
    return ppf


def interval_phi(classes_phi: ndarray) -> float:
    """
    Get the phi interval of grain size classes.

    :param classes_phi: The grain size classes in phi values.
    :return: The interval in phi values.
    """
    return abs((classes_phi[0] - classes_phi[-1]) / (len(classes_phi) - 1))


def to_phi(sizes: Union[int, float, ndarray]):
    """
    Convert the grain sizes from microns to phi values.

    :param sizes: The grain sizes in microns.
    :return: The grain sizes in phi values.
    """
    return -np.log2(sizes / 1000.0)


def to_microns(sizes_phi: ndarray):
    """
    Convert the grain sizes from phi values to microns.

    :param sizes_phi: The grain sizes in phi values.
    :return: The grain sizes in microns.
    """
    return np.exp2(-sizes_phi) * 1000.0


def cm(classes_phi: ndarray, distribution: ndarray) -> Tuple[float, float]:
    """
    Get the C and M values of a grain size distribution.
    C is the first percentile in phi scale, and M is the median in phi scale.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: A tuple that contains the C and M values.
    """
    ppf = reversed_phi_ppf(classes_phi, distribution)
    CM = ppf(0.99), ppf(0.5)
    return CM


# The following five formulas of calculating the statistical parameters referred to Blott & Pye (2001)'s work
# DOI: 10.1002/esp.261
def arithmetic(classes: ndarray, distribution: ndarray) -> Dict[str, float]:
    """
    Calculate the basic statistical parameters.
    Follow the "arithmetic method of moments" in Blott & Pye (2001).

    :param classes: The grain size classes in microns.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: A `dict` that contains the basic statistical parameters.
        `dict(mean=..., std=..., skewness=..., kurtosis=...)`
    """
    mean = np.sum(classes * distribution)
    std = np.sqrt(np.sum(distribution * (classes - mean) ** 2))
    skewness = np.sum(distribution * (classes - mean) ** 3) / (std ** 3)
    kurtosis = np.sum(distribution * (classes - mean) ** 4) / (std ** 4)
    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)


def geometric(classes: ndarray, distribution: ndarray) -> Dict[str, Union[float, str]]:
    """
    Calculate the basic statistical parameters.
    Follow the "geometric method of moments" in Blott & Pye (2001).

    :param classes: The grain size classes in microns.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: A `dict` that contains the basic statistical parameters and corresponding descriptions.
        `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=...,
            kurtosis_description=...)`
    """

    def std_description(std_value):
        if std_value < 1.27:
            return "Very well sorted"
        elif std_value < 1.41:
            return "Well sorted"
        elif std_value < 1.62:
            return "Moderately well sorted"
        elif std_value < 2.00:
            return "Moderately sorted"
        elif std_value < 4.00:
            return "Poorly sorted"
        elif std_value < 16.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness_value):
        if skewness_value < -1.30:
            return "Very fine skewed"
        elif skewness_value < -0.43:
            return "Fine skewed"
        elif skewness_value < 0.43:
            return "Symmetrical"
        elif skewness_value < 1.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis_value):
        if kurtosis_value < 1.70:
            return "Very platykurtic"
        elif kurtosis_value < 2.55:
            return "Platykurtic"
        elif kurtosis_value < 3.70:
            return "Mesokurtic"
        elif kurtosis_value < 7.40:
            return "Leptokurtic"
        else:
            return "Very leptokurtic"

    mean = np.exp(np.sum(distribution * np.log(classes)))
    std = np.exp(np.sqrt(np.sum(distribution * (np.log(classes) - np.log(mean)) ** 2)))
    skewness = np.sum(distribution * (np.log(classes) - np.log(mean)) ** 3) / (np.log(std) ** 3)
    kurtosis = np.sum(distribution * (np.log(classes) - np.log(mean)) ** 4) / (np.log(std) ** 4)

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))


def logarithmic(classes_phi: ndarray, distribution: ndarray) -> Dict[str, Union[float, str]]:
    """
    Calculate the basic statistical parameters.
    Follow the "logarithmic method of moments" in Blott & Pye (2001).

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: A `dict` that contains the basic statistical parameters and corresponding descriptions.
        `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=...,
            kurtosis_description=...)`
    """

    def std_description(std_value):
        if std_value < 0.35:
            return "Very well sorted"
        elif std_value < 0.50:
            return "Well sorted"
        elif std_value < 0.70:
            return "Moderately well sorted"
        elif std_value < 1.00:
            return "Moderately sorted"
        elif std_value < 2.00:
            return "Poorly sorted"
        elif std_value < 4.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness_value):
        skewness_value = -skewness_value
        if skewness_value < -1.30:
            return "Very fine skewed"
        elif skewness_value < -0.43:
            return "Fine skewed"
        elif skewness_value < 0.43:
            return "Symmetrical"
        elif skewness_value < 1.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis_value):
        if kurtosis_value < 1.70:
            return "Very platykurtic"
        elif kurtosis_value < 2.55:
            return "Platykurtic"
        elif kurtosis_value < 3.70:
            return "Mesokurtic"
        elif kurtosis_value < 7.40:
            return "Leptokurtic"
        else:
            return "Very leptokurtic"

    mean = np.sum(classes_phi * distribution)
    std = np.sqrt(np.sum(distribution * (classes_phi - mean) ** 2))
    skewness = np.sum(distribution * (classes_phi - mean) ** 3) / (std ** 3)
    kurtosis = np.sum(distribution * (classes_phi - mean) ** 4) / (std ** 4)

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))


def logarithmic_fw57(_ppf: Callable[[Union[int, float, np.ndarray]], Union[int, float, np.ndarray]]) -> \
        Dict[str, Union[float, str]]:
    """
    Calculate the basic statistical parameters.
    Follow the "logarithmic (original) Folk & Ward (1957) graphical measures" in Blott & Pye (2001).

    :param _ppf: A `interp1d` object returned by `reversed_phi_ppf` function.
    :return: A `dict` that contains the basic statistical parameters and corresponding descriptions.
        `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=...,
            kurtosis_description=...)`
    """

    def ppf(x):
        if not isinstance(x, ndarray):
            x = np.array(x)
        return _ppf(1 - x)

    def std_description(std_value):
        if std_value < 0.35:
            return "Very well sorted"
        elif std_value < 0.50:
            return "Well sorted"
        elif std_value < 0.70:
            return "Moderately well sorted"
        elif std_value < 1.00:
            return "Moderately sorted"
        elif std_value < 2.00:
            return "Poorly sorted"
        elif std_value < 4.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness_value):
        skewness_value = -skewness_value
        if skewness_value < -0.3:
            return "Very fine skewed"
        elif skewness_value < -0.1:
            return "Fine skewed"
        elif skewness_value < 0.1:
            return "Symmetrical"
        elif skewness_value < 0.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis_value):
        if kurtosis_value < 0.67:
            return "Very platykurtic"
        elif kurtosis_value < 0.90:
            return "Platykurtic"
        elif kurtosis_value < 1.11:
            return "Mesokurtic"
        elif kurtosis_value < 1.50:
            return "Leptokurtic"
        elif kurtosis_value < 3.00:
            return "Very leptokurtic"
        else:
            return "Extremely leptokurtic"

    mean = np.mean(ppf([0.16, 0.50, 0.84]))
    std = (ppf(0.84) - ppf(0.16)) / 4 + (ppf(0.95) - ppf(0.05)) / 6.6
    skewness = (ppf(0.16) + ppf(0.84) - 2 * ppf(0.50)) / 2 / (ppf(0.84) - ppf(0.16)) + (
            ppf(0.05) + ppf(0.95) - 2 * ppf(0.50)) / 2 / (ppf(0.95) - ppf(0.05))
    kurtosis = (ppf(0.95) - ppf(0.05)) / (2.44 * (ppf(0.75) - ppf(0.25)))

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))


def geometric_fw57(_ppf: Callable[[Union[int, float, np.ndarray]], Union[int, float, np.ndarray]]) -> \
        Dict[str, Union[float, str]]:
    """
    Calculate the basic statistical parameters.
    Follow the "geometric (modified) Folk & Ward (1957) graphical measures" in Blott & Pye (2001).

    :param _ppf: A `interp1d` object returned by `reversed_phi_ppf` function.
    :return: A `dict` that contains the basic statistical parameters and corresponding descriptions.
        `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=...,
            kurtosis_description=...)`
    """

    def log_ppf(x):
        if not isinstance(x, ndarray):
            x = np.array(x)
        return np.log(to_microns(_ppf(x)))

    def std_description(std_value):
        if std_value < 1.27:
            return "Very well sorted"
        elif std_value < 1.41:
            return "Well sorted"
        elif std_value < 1.62:
            return "Moderately well sorted"
        elif std_value < 2.00:
            return "Moderately sorted"
        elif std_value < 4.00:
            return "Poorly sorted"
        elif std_value < 16.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness_value):
        if skewness_value < -0.3:
            return "Very fine skewed"
        elif skewness_value < -0.1:
            return "Fine skewed"
        elif skewness_value < 0.1:
            return "Symmetrical"
        elif skewness_value < 0.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis_value):
        if kurtosis_value < 0.67:
            return "Very platykurtic"
        elif kurtosis_value < 0.90:
            return "Platykurtic"
        elif kurtosis_value < 1.11:
            return "Mesokurtic"
        elif kurtosis_value < 1.50:
            return "Leptokurtic"
        elif kurtosis_value < 3.00:
            return "Very leptokurtic"
        else:
            return "Extremely leptokurtic"

    mean = np.exp(np.mean(log_ppf([0.16, 0.50, 0.84])))
    std = np.exp((log_ppf(0.84) - log_ppf(0.16)) / 4 + (log_ppf(0.95) - log_ppf(0.05)) / 6.6)
    skewness = (log_ppf(0.16) + log_ppf(0.84) - 2 * log_ppf(0.50)) / 2 / (log_ppf(0.84) - log_ppf(0.16)) + (
            log_ppf(0.05) + log_ppf(0.95) - 2 * log_ppf(0.50)) / 2 / (log_ppf(0.95) - log_ppf(0.05))
    kurtosis = (log_ppf(0.95) - log_ppf(0.05)) / (2.44 * (log_ppf(0.75) - log_ppf(0.25)))

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))


# Referred to Blott & Pye (2012)'s grain size scale
# DOI: 10.1111/j.1365-3091.2012.01335.x
def scale_description(phi: Union[int, float]) -> Tuple[str, str]:
    """
    Get the size description of a φ value. Follow Blott & Pye (2012)'s grain size scale.

    :param phi: The phi value of grain size.
    :returns:
        adj: The adjective of a grade, e.g., "Very large".
        grade: The name of this grade, e.g., "Boulder".
    """
    if phi < -11:
        return "", "Megaclasts"
    if phi < -10:
        return "Very large", "Boulder"
    elif phi < -9:
        return "Large", "Boulder"
    elif phi < -8:
        return "Medium", "Boulder"
    elif phi < -7:
        return "Small", "Boulder"
    elif phi < -6:
        return "Very small", "Boulder"
    elif phi < -5:
        return "Very coarse", "Gravel"
    elif phi < -4:
        return "Coarse", "Gravel"
    elif phi < -3:
        return "Medium", "Gravel"
    elif phi < -2:
        return "Fine", "Gravel"
    elif phi < -1:
        return "Very fine", "Gravel"
    elif phi < 0:
        return "Very coarse", "Sand"
    elif phi < 1:
        return "Coarse", "Sand"
    elif phi < 2:
        return "Medium", "Sand"
    elif phi < 3:
        return "Fine", "Sand"
    elif phi < 4:
        return "Very fine", "Sand"
    elif phi < 5:
        return "Very coarse", "Silt"
    elif phi < 6:
        return "Coarse", "Silt"
    elif phi < 7:
        return "Medium", "Silt"
    elif phi < 8:
        return "Fine", "Silt"
    elif phi < 9:
        return "Very fine", "Silt"
    elif phi < 10:
        return "Very coarse", "Clay"
    elif phi < 11:
        return "Coarse", "Clay"
    elif phi < 12:
        return "Medium", "Clay"
    elif phi < 13:
        return "Fine", "Clay"
    else:
        return "Very fine", "Clay"


def _all_scales():
    return tuple([scale_description(phi) for phi in np.linspace(-11.5, 13.5, 26)])


# GSM: Gravel, Sand, Mud
def proportions_gsm(classes_phi: ndarray, distribution: ndarray) -> \
        Tuple[float, float, float]:
    """
    Get the proportions `[0, 1]` of gravel, sand and mud.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :returns:
        g: The proportion of gravel (-6 <= φ < -1).
        s: The proportion of sand (-1 <= φ < 4).
        m: The proportion of mud (φ >= 4).
    """
    gravel = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, -6), np.less(classes_phi, -1)], axis=0)]))
    sand = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, -1), np.less(classes_phi, 4)], axis=0)]))
    mud = np.max(np.sum(distribution[np.greater_equal(classes_phi, 4)]))
    return gravel, sand, mud


# SSC: Sand, Silt, Clay
def proportions_ssc(classes_phi: ndarray, distribution: ndarray) -> \
        Tuple[float, float, float]:
    """
    Get the proportions `[0, 1]` of sand, silt and clay.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :returns:
        s: The proportion of sand (-1 <= φ < 4).
        si: The proportion of silt (4 <= φ < 9).
        c: The proportion of clay (φ >= 9).
    """
    sand = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, -1), np.less(classes_phi, 4)], axis=0)]))
    silt = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, 4), np.less(classes_phi, 9)], axis=0)]))
    clay = np.max(np.sum(distribution[np.greater_equal(classes_phi, 9)]))
    return sand, silt, clay


# BGSSC: Boulder, Gravel, Sand, Silt, Clay
def proportions_bgssc(classes_phi: ndarray, distribution: ndarray) -> \
        Tuple[float, float, float, float, float]:
    """
    Get the proportions `[0, 1]` of boulder, gravel, sand, silt and clay.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :returns:
        b: The proportion of boulder (φ < -6).
        g: The proportion of gravel (-6 <= φ < -1).
        s: The proportion of sand (-1 <= φ < 4).
        si: The proportion of silt (4 <= φ < 9).
        c:  The proportion of clay (φ >= 9).
    """
    boulder = np.max(np.sum(distribution[np.less(classes_phi, -6)]))
    gravel = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, -6), np.less(classes_phi, -1)], axis=0)]))
    sand = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, -1), np.less(classes_phi, 4)], axis=0)]))
    silt = np.max(np.sum(distribution[np.all([np.greater_equal(classes_phi, 4), np.less(classes_phi, 9)], axis=0)]))
    clay = np.max(np.sum(distribution[np.greater_equal(classes_phi, 9)]))
    return boulder, gravel, sand, silt, clay


def all_proportions(classes_phi: ndarray, distribution: ndarray):
    """
    Get the proportions `[0, 1]` of all grades in `scale_description`.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: The proportions of all grades.
    """
    proportions = dict()
    all_scales = _all_scales()
    for scale in all_scales:
        proportions[scale] = 0.0
    for phi, freq in zip(classes_phi, distribution):
        scale = scale_description(phi)
        proportions[scale] += freq
    return proportions


def group_gsm_folk54(gravel: float, sand: float, mud: float, trace=0.01):
    """
    Get the classification group of the gravel-sand-mud scheme of Folk (1954).

    :param gravel: The proportion of gravel (-6 <= φ < -1).
    :param sand: The proportion of sand (-1 <= φ < 4).
    :param mud: The proportion of mud (φ >= 4).
    :param trace: The `trace` is used to determine if it's `Slightly Gravelly`.
    :return: The name of this group.
    """
    if mud == 0.0:
        ratio = np.inf
    else:
        ratio = sand / mud

    if gravel < trace:
        if ratio < 1 / 9:
            return "Mud"
        elif ratio < 1:
            return "Sandy Mud"
        elif ratio < 9:
            return "Muddy Sand"
        else:
            return "Sand"
    elif gravel < 0.05:
        if ratio < 1 / 9:
            return "Slightly Gravelly Mud"
        elif ratio < 1:
            return "Slightly Gravelly Sandy Mud"
        elif ratio < 9:
            return "Slightly Gravelly Muddy Sand"
        else:
            return "Slightly Gravelly Sand"
    elif gravel < 0.3:
        if ratio < 1:
            return "Gravelly Mud"
        elif ratio < 9:
            return "Gravelly Muddy Sand"
        else:
            return "Gravelly Sand"
    elif gravel < 0.8:
        if ratio < 1:
            return "Muddy Gravel"
        elif ratio < 9:
            return "Muddy Sandy Gravel"
        else:
            return "Sandy Gravel"
    else:
        return "Gravel"


def group_ssc_folk54(sand: float, silt: float, clay: float):
    """
    Get the classification group of the sand-silt-clay scheme of Folk (1954).

    :param sand: The proportion of sand (-1 <= φ < 4).
    :param silt: The proportion of silt (4 <= φ < 9).
    :param clay: The proportion of clay (φ >= 9).
    :return: The name of this group.
    """
    if clay == 0.0:
        ratio = np.inf
    else:
        ratio = silt / clay
    if sand < 0.1:
        if ratio < 1 / 3:
            return "Clay"
        elif ratio < 2 / 3:
            return "Mud"
        else:
            return "Slit"
    elif sand < 0.5:
        if ratio < 1 / 3:
            return "Sandy Clay"
        elif ratio < 2 / 3:
            return "Sandy Mud"
        else:
            return "Sandy Slit"
    elif sand < 0.9:
        if ratio < 1 / 3:
            return "Clayey Sand"
        elif ratio < 2 / 3:
            return "Muddy Sand"
        else:
            return "Silty Sand"
    else:
        return "Sand"


def group_folk54(classes_phi: ndarray, distribution: ndarray):
    """
    Get the classification group of Folk (1954)'s scheme.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: The name of this group.
    """
    gravel, sand, mud = proportions_gsm(classes_phi, distribution)
    if gravel > 0.0:
        return group_gsm_folk54(gravel, sand, mud)
    else:
        sand, silt, clay = proportions_ssc(classes_phi, distribution)
        return group_ssc_folk54(sand, silt, clay)


# map the symbol to its corresponding description
GROUP_BP12_SYMBOL_MAP = {
    "G": "Gravel",
    "S": "Sand",
    "SI": "Silt",
    "M": "Mud",
    "C": "Clay",
    "g": "Gravelly",
    "s": "Sandy",
    "si": "Silty",
    "m": "Muddy",
    "c": "Clayey",
    "(g)": "Slightly gravelly",
    "(s)": "Slightly sandy",
    "(si)": "Slightly silty",
    "(m)": "Slightly muddy",
    "(c)": "Slightly clayey",
    "(vg)": "Very slightly gravelly",
    "(vs)": "Very slightly sandy",
    "(vsi)": "Very slightly silty",
    "(vm)": "Very slightly muddy",
    "(vc)": "Very slightly clayey"}


def group_gsm_bp12(gravel: float, sand: float, mud: float, trace=0.01) \
        -> Tuple[List[str], List[str]]:
    """
    Get the classification group of the gravel-sand-mud scheme of Blott & Pye (2012).

    :param gravel: The proportion of gravel (-6 <= φ < -1).
    :param sand: The proportion of sand (-1 <= φ < 4).
    :param mud: The proportion of mud (φ >= 4).
    :param trace: The `trace` is used to determine if it's `Very slightly` or `Slightly`.
    :returns:
        symbols: The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
        descriptions: The descriptions of this group.
            Use `string.capwords(" ".join(descriptions))` to get the complete description.
    """
    tags = {"vs": [], "s": [], "adj": [], "n": []}
    if gravel < trace:
        pass
    elif gravel < 0.05:
        tags["vs"].append("(vg)")
    elif gravel < 0.2:
        tags["s"].append("(g)")
    elif gravel < max(sand, mud):
        tags["adj"].append("g")
    else:
        tags["n"].append("G")

    if sand < trace:
        pass
    elif sand < 0.05:
        tags["vs"].append("(vs)")
    elif sand < 0.2:
        tags["s"].append("(s)")
    elif sand < max(gravel, mud):
        tags["adj"].append("s")
    else:
        tags["n"].append("S")

    if mud < trace:
        pass
    elif mud < 0.05:
        tags["vs"].append("(vm)")
    elif mud < 0.2:
        tags["s"].append("(m)")
    elif mud < max(gravel, sand):
        tags["adj"].append("m")
    else:
        tags["n"].append("M")

    symbols = tags["vs"] + tags["s"] + tags["adj"] + tags["n"]
    description = [GROUP_BP12_SYMBOL_MAP[s] for s in symbols]
    return symbols, description


def group_ssc_bp12(sand: float, silt: float, clay: float, trace=0.01) \
        -> Tuple[List[str], List[str]]:
    """
    Get the classification group of the sand-silt-clay scheme of Blott & Pye (2012).

    :param sand: The proportion of sand (-1 <= φ < 4).
    :param silt: The proportion of silt (4 <= φ < 9).
    :param clay: The proportion of clay (φ >= 9).
    :param trace: The `trace` is used to determine if it's `Very slightly` or `Slightly`.
    :returns:
        symbols: The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
        descriptions: The descriptions of this group.
            Use `string.capwords(" ".join(descriptions))` to get the complete description.
    """
    tags = {"vs": [], "s": [], "adj": [], "n": []}
    if sand < trace:
        pass
    elif sand < 0.05:
        tags["vs"].append("(vs)")
    elif sand < 0.2:
        tags["s"].append("(s)")
    elif sand < max(sand, clay):
        tags["adj"].append("s")
    else:
        tags["n"].append("S")

    if silt < trace:
        pass
    elif silt < 0.05:
        tags["vs"].append("(vsi)")
    elif silt < 0.2:
        tags["s"].append("(si)")
    elif silt < max(sand, clay):
        tags["adj"].append("si")
    else:
        tags["n"].append("SI")

    if clay < trace:
        pass
    elif clay < 0.05:
        tags["vs"].append("(vc)")
    elif clay < 0.2:
        tags["s"].append("(c)")
    elif clay < max(sand, silt):
        tags["adj"].append("c")
    else:
        tags["n"].append("C")

    symbols = tags["vs"] + tags["s"] + tags["adj"] + tags["n"]
    description = [GROUP_BP12_SYMBOL_MAP[s] for s in symbols]
    return symbols, description


def group_bp12(classes_phi: ndarray, distribution: ndarray) \
        -> Tuple[List[str], List[str]]:
    """
    Get the classification group of Blott & Pye (2012)'s scheme.

    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :returns:
        symbols: The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
        descriptions: The descriptions of this group.
            Use `string.capwords(" ".join(descriptions))` to get the complete description.
    """
    gravel, sand, mud = proportions_gsm(classes_phi, distribution)
    if gravel > 0.0:
        return group_gsm_bp12(gravel, sand, mud)
    else:
        sand, silt, clay = proportions_ssc(classes_phi, distribution)
        return group_ssc_bp12(sand, silt, clay)


def major_statistics(classes: ndarray, classes_phi: ndarray,
                     distribution: ndarray, is_geometric=False, is_fw57=False) -> dict:
    """
    Get the statistical parameters of a grain size distribution.

    :param classes: The grain size classes in microns.
    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :param is_geometric: If `True`, the grain sizes are in microns, else they are in phi values.
    :param is_fw57: If `True`, it will use Folk & Ward (1957)'s graphical method.
        Else, it will use the method of statistical moments.
    :return: The `dict` contains these statistical parameters, including
        `mean`, `std`, `skewness`, `kurtosis`,
        `mode`, `modes`, `median`, `mean_description`, `std_description`,
        `skewness_description`, `kurtosis_description`.
    """
    reverse_phi_ppf = reversed_phi_ppf(classes_phi, distribution)
    median_phi = np.max(reverse_phi_ppf(0.5))
    statistics = {}
    if is_geometric:
        if is_fw57:
            statistics.update(geometric_fw57(reverse_phi_ppf))
        else:
            statistics.update(geometric(classes, distribution))
        mean_phi = to_phi(statistics["mean"])
        statistics["median"] = to_microns(median_phi)
    else:
        if is_fw57:
            statistics = logarithmic_fw57(reverse_phi_ppf)
        else:
            statistics = logarithmic(classes_phi, distribution)
        mean_phi = statistics["mean"]
        statistics["median"] = median_phi
    mean_description = string.capwords(" ".join(scale_description(mean_phi))).strip()
    statistics["mean_description"] = mean_description
    statistics["mode"] = mode(classes, classes_phi, distribution, is_geometric=is_geometric)
    statistics["modes"] = modes(classes, classes_phi, distribution, is_geometric=is_geometric)[0]
    return statistics


def all_statistics(classes: ndarray, classes_phi: ndarray, distribution: ndarray):
    """
    Get the statistical parameters and classification groups of all methods of a grain size distribution.

    :param classes: The grain size classes in microns.
    :param classes_phi: The grain size classes in phi values.
    :param distribution: The frequency distribution of grain size classes.
        Note, the sum of frequencies should be equal to 1.
    :return: The `dict` contains the statistical parameters of all methods, including
        `arithmetic`, `geometric`, `logarithmic`, `geometric_fw57`, `logarithmic_fw57`;
        and the proportions of different grades, including `proportions_gsm`, `proportions_ssc`,
        `proportions_bgssc`, `proportions`; and the classification groups,
        including `group_folk54` and `group_bp12`.
    """
    result = {
        "arithmetic": arithmetic(classes, distribution),
        "geometric": major_statistics(classes, classes_phi, distribution, is_geometric=True, is_fw57=False),
        "logarithmic": major_statistics(classes, classes_phi, distribution, is_geometric=False, is_fw57=False),
        "geometric_fw57": major_statistics(classes, classes_phi, distribution, is_geometric=True, is_fw57=True),
        "logarithmic_fw57": major_statistics(classes, classes_phi, distribution, is_geometric=False, is_fw57=True),
        "proportions_gsm": proportions_gsm(classes_phi, distribution),
        "proportions_ssc": proportions_ssc(classes_phi, distribution),
        "proportions_bgssc": proportions_bgssc(classes_phi, distribution),
        "proportions": all_proportions(classes_phi, distribution),
        "group_folk54": group_folk54(classes_phi, distribution)}

    bp12_symbols, bp12_descriptions = group_bp12(classes_phi, distribution)
    bp12_symbol = "".join(bp12_symbols)
    bp12_description = string.capwords(" ".join(bp12_descriptions))
    result["_group_bp12_symbols"] = bp12_symbols
    result["group_bp12_symbol"] = bp12_symbol
    result["group_bp12"] = bp12_description
    return result
