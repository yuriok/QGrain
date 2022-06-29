import string
import typing

import numpy as np
from QGrain.statistics._base import *


# The following five formulas of calculating the statistical parameters referred to Blott & Pye (2001)'s work
# DOI: 10.1002/esp.261
def arithmetic(classes_μm: np.ndarray, frequency: np.ndarray) -> dict:
    """
    Calculate the basic statistical parameters, follow the "arithmetic method of moments" in Blott & Pye (2001).

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    statistics: `dict(mean=..., std=..., skewness=..., kurtosis=...)`
        The `dict` which contains the basic statistical parameters.

    """
    mean = np.sum(classes_μm * frequency)
    std = np.sqrt(np.sum(frequency * (classes_μm-mean)**2))
    skewness = np.sum(frequency * (classes_μm-mean)**3) / (std ** 3)
    kurtosis = np.sum(frequency * (classes_μm-mean)**4) / (std ** 4)
    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)

def geometric(classes_μm: np.ndarray, frequency: np.ndarray) -> dict:
    """
    Calculate the basic statistical parameters, follow the "geometric method of moments" in Blott & Pye (2001).

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    statistics: `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=..., kurtosis_description=...)`
        The `dict` which contains the basic statistical parameters and corresponding descriptions.

    """
    mean = np.exp(np.sum(frequency * np.log(classes_μm)))
    std = np.exp(np.sqrt(np.sum(frequency * (np.log(classes_μm) - np.log(mean))**2)))
    skewness = np.sum(frequency * (np.log(classes_μm) - np.log(mean))**3) / (np.log(std) ** 3)
    kurtosis = np.sum(frequency * (np.log(classes_μm) - np.log(mean))**4) / (np.log(std) ** 4)

    def std_description(std):
        if std < 1.27:
            return "Very well sorted"
        elif std < 1.41:
            return "Well sorted"
        elif std < 1.62:
            return "Moderately well sorted"
        elif std < 2.00:
            return "Moderately sorted"
        elif std < 4.00:
            return "Poorly sorted"
        elif std < 16.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness):
        if skewness < -1.30:
            return "Very fine skewed"
        elif skewness < -0.43:
            return "Fine skewed"
        elif skewness < 0.43:
            return "Symmetrical"
        elif skewness < 1.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis):
        if kurtosis < 1.70:
            return "Very platykurtic"
        elif kurtosis < 2.55:
            return "Platykurtic"
        elif kurtosis < 3.70:
            return "Mesokurtic"
        elif kurtosis < 7.40:
            return "Leptokurtic"
        else:
            return "Very leptokurtic"

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))

def logarithmic(classes_φ: np.ndarray, frequency: np.ndarray) -> dict:
    """
    Calculate the basic statistical parameters, follow the "logarithmic method of moments" in Blott & Pye (2001).

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    statistics: `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=..., kurtosis_description=...)`
        The `dict` which contains the basic statistical parameters and corresponding descriptions.

    """
    mean = np.sum(classes_φ * frequency)
    std = np.sqrt(np.sum(frequency * (classes_φ-mean)**2))
    skewness = np.sum(frequency * (classes_φ-mean)**3) / (std ** 3)
    kurtosis = np.sum(frequency * (classes_φ-mean)**4) / (std ** 4)
    def std_description(std):
        if std < 0.35:
            return "Very well sorted"
        elif std < 0.50:
            return "Well sorted"
        elif std < 0.70:
            return "Moderately well sorted"
        elif std < 1.00:
            return "Moderately sorted"
        elif std < 2.00:
            return "Poorly sorted"
        elif std < 4.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness):
        skewness = -skewness
        if skewness < -1.30:
            return "Very fine skewed"
        elif skewness < -0.43:
            return "Fine skewed"
        elif skewness < 0.43:
            return "Symmetrical"
        elif skewness < 1.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis):
        if kurtosis < 1.70:
            return "Very platykurtic"
        elif kurtosis < 2.55:
            return "Platykurtic"
        elif kurtosis < 3.70:
            return "Mesokurtic"
        elif kurtosis < 7.40:
            return "Leptokurtic"
        else:
            return "Very leptokurtic"

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))

def logarithmic_FW57(reverse_phi_ppf) -> dict:
    """
    Calculate the basic statistical parameters, follow the "logarithmic (original) Folk & Ward (1957) graphical measures" in Blott & Pye (2001).

    ## Parameters

    reverse_phi_ppf: `scipy.interpolate.interp1d`
        A `interp1d` object returned by `get_reverse_phi_ppf` function in `QGrain.statistics._base`

    ## Returns

    statistics: `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=..., kurtosis_description=...)`
        The `dict` which contains the basic statistical parameters and corresponding descriptions.

    """
    _ppf = reverse_phi_ppf
    def ppf(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return _ppf(1-x)

    mean = np.mean(ppf([0.16, 0.50, 0.84]))
    std = (ppf(0.84)-ppf(0.16)) / 4 + (ppf(0.95)-ppf(0.05)) / 6.6
    skewness = (ppf(0.16)+ppf(0.84)-2*ppf(0.50)) / 2 / (ppf(0.84) - ppf(0.16)) + (ppf(0.05)+ppf(0.95)-2*ppf(0.50)) / 2 / (ppf(0.95)-ppf(0.05))
    kurtosis = (ppf(0.95)-ppf(0.05)) / (2.44*(ppf(0.75)-ppf(0.25)))
    def std_description(std):
        if std < 0.35:
            return "Very well sorted"
        elif std < 0.50:
            return "Well sorted"
        elif std < 0.70:
            return "Moderately well sorted"
        elif std < 1.00:
            return "Moderately sorted"
        elif std < 2.00:
            return "Poorly sorted"
        elif std < 4.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness):
        skewness = -skewness
        if skewness < -0.3:
            return "Very fine skewed"
        elif skewness < -0.1:
            return "Fine skewed"
        elif skewness < 0.1:
            return "Symmetrical"
        elif skewness < 0.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis):
        if kurtosis < 0.67:
            return "Very platykurtic"
        elif kurtosis < 0.90:
            return "Platykurtic"
        elif kurtosis < 1.11:
            return "Mesokurtic"
        elif kurtosis < 1.50:
            return "Leptokurtic"
        elif kurtosis < 3.00:
            return "Very leptokurtic"
        else:
            return "Extremely leptokurtic"

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))

def geometric_FW57(reverse_phi_ppf) -> dict:
    """
    Calculate the basic statistical parameters, follow the "geometric (modified) Folk & Ward (1957) graphical measures" in Blott & Pye (2001).

    ## Parameters

    reverse_phi_ppf: `scipy.interpolate.interp1d`
        A `interp1d` object returned by `get_reverse_phi_ppf` function in `QGrain.statistics._base`

    ## Returns

    statistics: `dict(mean=..., std=..., skewness=..., kurtosis=..., std_description=..., skewness_description=..., kurtosis_description=...)`
        The `dict` which contains the basic statistical parameters and corresponding descriptions.

    """
    _ppf = reverse_phi_ppf
    def log_ppf(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return np.log(convert_φ_to_μm(_ppf(x)))

    mean = np.exp(np.mean(log_ppf([0.16, 0.50, 0.84])))
    std = np.exp((log_ppf(0.84)-log_ppf(0.16)) / 4 + (log_ppf(0.95)-log_ppf(0.05)) / 6.6)
    skewness = (log_ppf(0.16)+log_ppf(0.84)-2*log_ppf(0.50)) / 2 / (log_ppf(0.84) - log_ppf(0.16)) + (log_ppf(0.05)+log_ppf(0.95)-2*log_ppf(0.50)) / 2 / (log_ppf(0.95)-log_ppf(0.05))
    kurtosis = (log_ppf(0.95)-log_ppf(0.05)) / (2.44*(log_ppf(0.75)-log_ppf(0.25)))
    def std_description(std):
        if std < 1.27:
            return "Very well sorted"
        elif std < 1.41:
            return "Well sorted"
        elif std < 1.62:
            return "Moderately well sorted"
        elif std < 2.00:
            return "Moderately sorted"
        elif std < 4.00:
            return "Poorly sorted"
        elif std < 16.00:
            return "Very poorly sorted"
        else:
            return "Extremely poorly sorted"

    def skewness_description(skewness):
        if skewness < -0.3:
            return "Very fine skewed"
        elif skewness < -0.1:
            return "Fine skewed"
        elif skewness < 0.1:
            return "Symmetrical"
        elif skewness < 0.30:
            return "Coarse skewed"
        else:
            return "Very coarse skewed"

    def kurtosis_description(kurtosis):
        if kurtosis < 0.67:
            return "Very platykurtic"
        elif kurtosis < 0.90:
            return "Platykurtic"
        elif kurtosis < 1.11:
            return "Mesokurtic"
        elif kurtosis < 1.50:
            return "Leptokurtic"
        elif kurtosis < 3.00:
            return "Very leptokurtic"
        else:
            return "Extremely leptokurtic"

    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis,
                std_description=std_description(std),
                skewness_description=skewness_description(skewness),
                kurtosis_description=kurtosis_description(kurtosis))

# Referred to Blott & Pye (2012)'s grain size scale
# DOI: 10.1111/j.1365-3091.2012.01335.x
def scale_description(φ) -> typing.Tuple[str, str]:
    """
    Get the size description of a φ value. Follow Blott & Pye (2012)'s grain size scale.

    ## Parameters

    φ: `float`
        The φ value of grain size.

    ## Returns

    adj: `str`
        The adjective of a grade, e.g., "Very large".
    grade: `str`
        The name of this grade, e.g., "Boulder".

    """
    if φ < -11:
        return "", "Megaclasts"
    if φ < -10:
        return "Very large", "Boulder"
    elif φ < -9:
        return "Large", "Boulder"
    elif φ < -8:
        return "Medium", "Boulder"
    elif φ < -7:
        return "Small", "Boulder"
    elif φ < -6:
        return "Very small", "Boulder"
    elif φ < -5:
        return "Very coarse", "Gravel"
    elif φ < -4:
        return "Coarse", "Gravel"
    elif φ < -3:
        return "Medium", "Gravel"
    elif φ < -2:
        return "Fine", "Gravel"
    elif φ < -1:
        return "Very fine", "Gravel"
    elif φ < 0:
        return "Very coarse", "Sand"
    elif φ < 1:
        return "Coarse", "Sand"
    elif φ < 2:
        return "Medium", "Sand"
    elif φ < 3:
        return "Fine", "Sand"
    elif φ < 4:
        return "Very fine", "Sand"
    elif φ < 5:
        return "Very coarse", "Silt"
    elif φ < 6:
        return "Coarse", "Silt"
    elif φ < 7:
        return "Medium", "Silt"
    elif φ < 8:
        return "Fine", "Silt"
    elif φ < 9:
        return "Very fine", "Silt"
    elif φ < 10:
        return "Very coarse", "Clay"
    elif φ < 11:
        return "Coarse", "Clay"
    elif φ < 12:
        return "Medium", "Clay"
    elif φ < 13:
        return "Fine", "Clay"
    else:
        return "Very fine", "Clay"

def _get_all_scales():
    return tuple([scale_description(φ) for φ in np.linspace(-11.5, 13.5, 26)])

# GSM: Gravel, Sand, Mud
def get_GSM_proportion(classes_φ: np.ndarray, frequency: np.ndarray) -> \
    typing.Tuple[float, float, float]:
    """
    Get the proportions `[0, 1]` of gravel, sand and mud.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    g: `float`
        The proportion of gravel (-6 <= φ < -1).
    s: `float`
        The proportion of sand (-1 <= φ < 4).
    m: `float`
        The proportion of mud (φ >= 4).

    """
    gravel = np.sum(frequency[np.all([np.greater_equal(classes_φ, -6), np.less(classes_φ, -1)], axis=0)])
    sand = np.sum(frequency[np.all([np.greater_equal(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    mud = np.sum(frequency[np.greater_equal(classes_φ, 4)])
    return gravel, sand, mud

# SSC: Sand, Silt, Clay
def get_SSC_proportion(classes_φ: np.ndarray, frequency: np.ndarray) -> \
    typing.Tuple[float, float, float]:
    """
    Get the proportions `[0, 1]` of sand, silt and clay.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    s: `float`
        The proportion of sand (-1 <= φ < 4).
    si: `float`
        The proportion of silt (4 <= φ < 9).
    c: `float`
        The proportion of clay (φ >= 9).

    """
    sand = np.sum(frequency[np.all([np.greater_equal(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    silt = np.sum(frequency[np.all([np.greater_equal(classes_φ, 4), np.less(classes_φ, 9)], axis=0)])
    clay = np.sum(frequency[np.greater_equal(classes_φ, 9)])
    return sand, silt, clay

# BGSSC: Boulder, Gravel, Sand, Silt, Clay
def get_BGSSC_proportion(classes_φ: np.ndarray, frequency: np.ndarray) -> \
    typing.Tuple[float, float, float, float, float]:
    """
    Get the proportions `[0, 1]` of boulder, gravel, sand, silt and clay.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    b: `float`
        The proportion of boulder (φ < -6).
    g: `float`
        The proportion of gravel (-6 <= φ < -1).
    s: `float`
        The proportion of sand (-1 <= φ < 4).
    si: `float`
        The proportion of silt (4 <= φ < 9).
    c: `float`
        The proportion of clay (φ >= 9).

    """
    boulder = np.sum(frequency[np.less(classes_φ, -6)])
    gravel = np.sum(frequency[np.all([np.greater_equal(classes_φ, -6), np.less(classes_φ, -1)], axis=0)])
    sand = np.sum(frequency[np.all([np.greater_equal(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    silt = np.sum(frequency[np.all([np.greater_equal(classes_φ, 4), np.less(classes_φ, 9)], axis=0)])
    clay = np.sum(frequency[np.greater_equal(classes_φ, 9)])
    return boulder, gravel, sand, silt, clay

def get_all_proportion(classes_φ: np.ndarray, frequency: np.ndarray):
    """
    Get the proportions `[0, 1]` of all grades in `scale_description`.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    proportion: `dict`
        The proportion of all grades.

    """
    proporations = dict()
    all_scales = _get_all_scales()
    for scale in all_scales:
        proporations[scale] = 0.0
    for φ, freq in zip(classes_φ, frequency):
        scale = scale_description(φ)
        proporations[scale] += freq
    return proporations

def get_GSM_group_Folk54(gravel: float, sand: float, mud: float, trace=0.01):
    """
    Get the classification group of the gravel-sand-mud scheme of Folk (1954).

    ## Parameters

    gravel: `float`
        The proportion of gravel (-6 <= φ < -1).
    sand: `float`
        The proportion of sand (-1 <= φ < 4).
    mud: `float`
        The proportion of mud (φ >= 4).
    trace: `float` (default `0.01` or `1%`)
        The `trace` is used to determine if it's `Slightly Gravelly`.

    ## Returns

    group: `str`
        The name of this group.

    """
    if mud == 0.0:
        ratio = np.inf
    else:
        ratio = sand / mud

    if gravel < trace:
        if ratio < 1/9:
            return "Mud"
        elif ratio < 1:
            return "Sandy Mud"
        elif ratio < 9:
            return "Muddy Sand"
        else:
            return "Sand"
    elif gravel < 0.05:
        if ratio < 1/9:
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

def get_SSC_group_Folk54(sand: float, silt: float, clay: float):
    """
    Get the classification group of the sand-silt-clay scheme of Folk (1954).

    ## Parameters

    sand: `float`
        The proportion of sand (-1 <= φ < 4).
    silt: `float`
        The proportion of silt (4 <= φ < 9).
    clay: `float`
        The proportion of clay (φ >= 9).

    ## Returns

    group: `str`
        The name of this group.

    """
    if clay == 0.0:
        ratio = np.inf
    else:
        ratio = silt / clay
    if sand < 0.1:
        if ratio < 1/3:
            return "Clay"
        elif ratio < 2/3:
            return "Mud"
        else:
            return "Slit"
    elif sand < 0.5:
        if ratio < 1/3:
            return "Sandy Clay"
        elif ratio < 2/3:
            return "Sandy Mud"
        else:
            return "Sandy Slit"
    elif sand < 0.9:
        if ratio < 1/3:
            return "Clayey Sand"
        elif ratio < 2/3:
            return "Muddy Sand"
        else:
            return "Slity Sand"
    else:
        return "Sand"

def get_group_Folk54(classes_φ: np.ndarray, frequency: np.ndarray):
    """
    Get the classification group of Folk (1954)'s scheme.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    group: `str`
        The name of this group.

    """
    gravel, sand, mud = get_GSM_proportion(classes_φ, frequency)
    if gravel > 0.0:
        return get_GSM_group_Folk54(gravel, sand, mud)
    else:
        sand, silt, clay = get_SSC_proportion(classes_φ, frequency)
        return get_SSC_group_Folk54(sand, silt, clay)

# map the symbol to its corresponding description
GROUP_BP12_SYMBOL_MAP = {
    "G":"Gravel",
    "S": "Sand",
    "SI": "Silt",
    "M":"Mud",
    "C": "Clay",
    "g":"Gravelly",
    "s": "Sandy",
    "si": "Silty",
    "m":"Muddy",
    "c": "Clayey",
    "(g)":"Slightly gravelly",
    "(s)": "Slightly sandy",
    "(si)": "Slightly silty",
    "(m)":"Slightly muddy",
    "(c)": "Slightly clayey",
    "(vg)":"Very slightly gravelly",
    "(vs)": "Very slightly sandy",
    "(vsi)": "Very slightly silty",
    "(vm)":"Very slightly muddy",
    "(vc)": "Very slightly clayey"}

def get_GSM_group_BP12(gravel: float, sand: float, mud: float, trace=0.01) \
    -> typing.Tuple[typing.List[str], typing.List[str]]:
    """
    Get the classification group of the gravel-sand-mud scheme of Blott & Pye (2012).

    ## Parameters

    gravel: `float`
        The proportion of gravel (-6 <= φ < -1).
    sand: `float`
        The proportion of sand (-1 <= φ < 4).
    mud: `float`
        The proportion of mud (φ >= 4).
    trace: `float` (default `0.01` or `1%`)
        The `trace` is used to determine that it's `Very slightly` or `Slightly`.

    ## Returns

    symbols: `list(str)`
        The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
    descriptions: `list(str)`
        The descriptions of this group. Use `string.capwords(" ".join(descriptions))` to get the whole description.

    """
    tags = {}
    tags["vs"] = []
    tags["s"] = []
    tags["adj"] = []
    tags["n"] = []
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

    symbols = tags["vs"]+tags["s"]+tags["adj"]+tags["n"]
    description = [GROUP_BP12_SYMBOL_MAP[s] for s in symbols]
    return symbols, description

def get_SSC_group_BP12(sand: float, silt: float, clay: float, trace=0.01) \
    -> typing.Tuple[typing.List[str], typing.List[str]]:
    """
    Get the classification group of the sand-silt-clay scheme of Blott & Pye (2012).

    ## Parameters

    sand: `float`
        The proportion of sand (-1 <= φ < 4).
    silt: `float`
        The proportion of silt (4 <= φ < 9).
    clay: `float`
        The proportion of clay (φ >= 9).
    trace: `float` (default `0.01` or `1%`)
        The `trace` is used to determine that it's `Very slightly` or `Slightly`.

    ## Returns

    symbols: `list(str)`
        The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
    descriptions: `list(str)`
        The descriptions of this group. Use `string.capwords(" ".join(descriptions))` to get the whole description.

    """
    tags = {}
    tags["vs"] = []
    tags["s"] = []
    tags["adj"] = []
    tags["n"] = []
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

    symbols = tags["vs"]+tags["s"]+tags["adj"]+tags["n"]
    description = [GROUP_BP12_SYMBOL_MAP[s] for s in symbols]
    return symbols, description

def get_group_BP12(classes_φ: np.ndarray, frequency: np.ndarray) \
    -> typing.Tuple[typing.List[str], typing.List[str]]:
    """
    Get the classification group of Blott & Pye (2012)'s scheme.

    ## Parameters

    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns

    symbols: `list(str)`
        The symbols of this group. Use `"".join(symbols)` to get the whole symbol.
    descriptions: `list(str)`
        The descriptions of this group. Use `string.capwords(" ".join(descriptions))` to get the whole description.

    """
    gravel, sand, mud = get_GSM_proportion(classes_φ, frequency)
    if gravel > 0.0:
        return get_GSM_group_BP12(gravel, sand, mud)
    else:
        sand, silt, clay = get_SSC_proportion(classes_φ, frequency)
        return get_SSC_group_BP12(sand, silt, clay)

def get_statistics(classes_μm: np.ndarray,
                  classes_φ: np.ndarray,
                  frequency: np.ndarray,
                  is_geometric=False, is_FW57=False) -> dict:
    """
    Get the statistical parameters of a grain size distribution.

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.
    is_geometric: `bool` (default `False`)
        If `True` the grain size unit is μm, else it's φ.
    is_FW57: `bool` (default `False`)
        If `True` it will use Folk & Ward (1957)'s graphical method, else it will use the method of statistical moments.

    ## Returns
    statistics: `dict`
        The `dict` contains these statistical parameters, including `mean`, `std`, `skewness`, `kurtosis`, `mode`, `modes`, `median`, `mean_description`, `std_description`, `skewness_description`, `kurtosis_description`.

    """
    reverse_phi_ppf = get_reverse_phi_ppf(classes_φ, frequency)
    median_φ = reverse_phi_ppf(0.5).max()
    if is_geometric:
        if is_FW57:
            statistics = geometric_FW57(reverse_phi_ppf)
        else:
            statistics = geometric(classes_μm, frequency)
        mean_φ = convert_μm_to_φ(statistics["mean"])
        statistics["median"] = convert_φ_to_μm(median_φ)
    else:
        if is_FW57:
            statistics = logarithmic_FW57(reverse_phi_ppf)
        else:
            statistics = logarithmic(classes_φ, frequency)
        mean_φ = statistics["mean"]
        statistics["median"] = median_φ
    mean_description = string.capwords(" ".join(scale_description(mean_φ))).strip()
    statistics["mean_description"] = mean_description
    statistics["mode"] = get_mode(classes_μm, classes_φ, frequency, is_geometric=is_geometric)
    statistics["modes"] = get_modes(classes_μm, classes_φ, frequency, is_geometric=is_geometric)[0]
    return statistics

def get_all_statistics(classes_μm: np.ndarray,
                      classes_φ: np.ndarray,
                      frequency: np.ndarray):
    """
    Get all statistical parameters and classification groups of a grain size distribution.

    ## Parameters

    classes_μm: `np.ndarray`
        The grain size classes in μm.
    classes_φ: `np.ndarray`
        The grain size classes in φ.
    frequency: `np.ndarray`
        The frequency values of grain size classes. Note that the sum of frequencies should be `1.0`.

    ## Returns
    all_statistics: `dict`
        The `dict` contains these statistical parameters all methods, including `arithmetic`, `geometric`, `logarithmic`, `geometric_FW57`, `logarithmic_FW57`; and the proportions of different grades, including `GSM_proportion`, `SSC_proportion`, `BGSSC_proportion`, `proportion`; and the classification groups, including `group_Folk54` and `group_BP12`.

    """
    result = {}
    result["arithmetic"] = arithmetic(classes_μm, frequency)
    result["geometric"] = get_statistics(classes_μm, classes_φ, frequency, is_geometric=True, is_FW57=False)
    result["logarithmic"] = get_statistics(classes_μm, classes_φ, frequency, is_geometric=False, is_FW57=False)
    result["geometric_FW57"] = get_statistics(classes_μm, classes_φ, frequency, is_geometric=True, is_FW57=True)
    result["logarithmic_FW57"] = get_statistics(classes_μm, classes_φ, frequency, is_geometric=False, is_FW57=True)
    result["GSM_proportion"] = get_GSM_proportion(classes_φ, frequency)
    result["SSC_proportion"] = get_SSC_proportion(classes_φ, frequency)
    result["BGSSC_proportion"] = get_BGSSC_proportion(classes_φ, frequency)
    result["proportion"] = get_all_proportion(classes_φ, frequency)

    result["group_Folk54"] = get_group_Folk54(classes_φ, frequency)
    BP12_symbol_strs, BP12_description_strs = get_group_BP12(classes_φ, frequency)
    BP12_symbol = "".join(BP12_symbol_strs)
    BP12_description = string.capwords(" ".join(BP12_description_strs))
    result["_group_BP12_symbols"] = BP12_symbol_strs
    result["group_BP12_symbol"] = BP12_symbol
    result["group_BP12"] = BP12_description

    return result
