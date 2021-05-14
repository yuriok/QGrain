import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import string
invalid_moments = dict(mean=np.nan, std=np.nan, skewness=np.nan, kurtosis=np.nan)

def get_modes(classes_μm, classes_φ, frequency, geometric=True):
    peaks, _ = find_peaks(frequency)
    position = classes_μm[peaks] if geometric else classes_φ[peaks]
    values = frequency[peaks]
    not_trace = np.greater(values, 0.01)
    return tuple(position[not_trace]), tuple(values[not_trace])

def get_cumulative_frequency(frequency, expand=False):
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

def get_ppf(classes_φ: np.ndarray, frequency: np.ndarray):
    interval = np.mean(classes_φ[1:] - classes_φ[:-1])
    expand_classes = np.linspace(classes_φ[0] - interval, classes_φ[-1]+interval, len(classes_φ)+2)
    cumulative_frequency = get_cumulative_frequency(frequency, expand=True)
    cumulative_frequency = np.array(cumulative_frequency)
    ppf = interp1d(cumulative_frequency, expand_classes, kind="slinear")
    return ppf

def convert_μm_to_φ(classes_μm: np.ndarray):
    return -np.log2(classes_μm / 1000.0)

def convert_φ_to_μm(classes_φ: np.ndarray):
    return np.exp2(-classes_φ) * 1000.0

# The following formulas of calculating the statistic parameters referred to Blott & Pye (2001)'s work
def arithmetic(classes_μm: np.ndarray, frequency: np.ndarray) -> dict:
    mean = np.sum(classes_μm * frequency)
    std = np.sqrt(np.sum(frequency * (classes_μm-mean)**2))
    skewness = np.sum(frequency * (classes_μm-mean)**3) / (std ** 3)
    kurtosis = np.sum(frequency * (classes_μm-mean)**4) / (std ** 4)
    return dict(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)

def geometric(classes_μm: np.ndarray, frequency: np.ndarray) -> dict:
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

# Referred to Blott & Pye (2012)'s grain-size scale
def scale_description(φ):
    if φ < -11:
        return "Megaclasts",
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

def get_all_scales():
    return tuple([scale_description(φ) for φ in np.linspace(-11.5, 13.5, 26)])

# GSM: Gravel, Sand, Mud
def get_GSM_proportion(classes_φ: np.ndarray, frequency: np.ndarray):
    gravel = np.sum(frequency[np.all([np.greater(classes_φ, -6), np.less(classes_φ, -1)], axis=0)])
    sand = np.sum(frequency[np.all([np.greater(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    mud = np.sum(frequency[np.greater(classes_φ, 4)])
    return gravel, sand, mud

# SSC: Sand, Silt, Clay
def get_SSC_proportion(classes_φ: np.ndarray, frequency: np.ndarray):
    sand = np.sum(frequency[np.all([np.greater(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    silt = np.sum(frequency[np.all([np.greater(classes_φ, 4), np.less(classes_φ, 9)], axis=0)])
    clay = np.sum(frequency[np.greater(classes_φ, 9)])
    return sand, silt, clay

# BGSSC: Boulder, Gravel, Sand, Silt, Clay
def get_BGSSC_proportion(classes_φ: np.ndarray, frequency: np.ndarray):
    boulder = np.sum(frequency[np.less(classes_φ, -6)])
    gravel = np.sum(frequency[np.all([np.greater(classes_φ, -6), np.less(classes_φ, -1)], axis=0)])
    sand = np.sum(frequency[np.all([np.greater(classes_φ, -1), np.less(classes_φ, 4)], axis=0)])
    silt = np.sum(frequency[np.all([np.greater(classes_φ, 4), np.less(classes_φ, 9)], axis=0)])
    clay = np.sum(frequency[np.greater(classes_φ, 9)])
    return boulder, gravel, sand, silt, clay

def get_all_proporation(classes_φ: np.ndarray, frequency: np.ndarray):
    proporations = dict()
    all_scales = get_all_scales()
    for scale in all_scales:
        proporations[scale] = 0.0
    for φ, freq in zip(classes_φ, frequency):
        scale = scale_description(φ)
        proporations[scale] += freq
    return proporations

def get_GSM_textural_group_Folk54(gravel: float, sand: float, mud: float):
    TRACE = 0.01
    if mud == 0.0:
        ratio = np.inf
    else:
        ratio = sand / mud

    if gravel < TRACE:
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

def get_SSC_textural_group_Folk54(sand: float, silt: float, clay: float):
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

def get_textural_group_Folk54(classes_φ: np.ndarray, frequency: np.ndarray):
    gravel, sand, mud = get_GSM_proportion(classes_φ, frequency)
    if gravel > 0.0:
        return get_GSM_textural_group_Folk54(gravel, sand, mud)
    else:
        sand, silt, clay = get_SSC_proportion(classes_φ, frequency)
        return get_SSC_textural_group_Folk54(sand, silt, clay)

def get_GSM_textural_group_BP12(gravel: float, sand: float, mud: float):
    symbol_map = {"G":"Gravel",
                  "S":"Sand",
                  "M":"Mud",
                  "g":"Gravelly",
                  "s":"Sandy",
                  "m":"Muddy",
                  "(g)":"Slightly gravelly",
                  "(s)":"Slightly sandy",
                  "(m)":"Slightly muddy",
                  "(vg)":"Very slightly gravelly",
                  "(vs)":"Very slightly sandy",
                  "(vm)":"Very slightly muddy"}

    TRACE = 0.01
    tags = {}
    tags["vs"] = []
    tags["s"] = []
    tags["adj"] = []
    tags["n"] = []
    if gravel < TRACE:
        pass
    elif gravel < 0.05:
        tags["vs"].append("(vg)")
    elif gravel < 0.2:
        tags["s"].append("(g)")
    elif gravel < max(sand, mud):
        tags["adj"].append("g")
    else:
        tags["n"].append("G")

    if sand < TRACE:
        pass
    elif sand < 0.05:
        tags["vs"].append("(vs)")
    elif sand < 0.2:
        tags["s"].append("(s)")
    elif sand < max(gravel, mud):
        tags["adj"].append("s")
    else:
        tags["n"].append("S")

    if mud < TRACE:
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
    description = [symbol_map[s] for s in symbols]
    return symbols, description

def get_SSC_textural_group_BP12(sand: float, silt: float, clay: float):
    symbol_map = {"S":"Sand",
                  "SI":"Silt",
                  "C":"Clay",
                  "s":"Sandy",
                  "si":"Silty",
                  "c":"Clayey",
                  "(s)":"Slightly sandy",
                  "(si)":"Slightly silty",
                  "(c)":"Slightly clayey",
                  "(vs)":"Very slightly sandy",
                  "(vsi)":"Very slightly silty",
                  "(vc)":"Very slightly clayey"}

    TRACE = 0.01
    tags = {}
    tags["vs"] = []
    tags["s"] = []
    tags["adj"] = []
    tags["n"] = []
    if sand < TRACE:
        pass
    elif sand < 0.05:
        tags["vs"].append("(vs)")
    elif sand < 0.2:
        tags["s"].append("(s)")
    elif sand < max(sand, clay):
        tags["adj"].append("s")
    else:
        tags["n"].append("S")

    if silt < TRACE:
        pass
    elif silt < 0.05:
        tags["vs"].append("(vsi)")
    elif silt < 0.2:
        tags["s"].append("(si)")
    elif silt < max(sand, clay):
        tags["adj"].append("si")
    else:
        tags["n"].append("SI")

    if clay < TRACE:
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
    description = [symbol_map[s] for s in symbols]
    return symbols, description

def get_textural_group_BP12(classes_φ: np.ndarray, frequency: np.ndarray):
    gravel, sand, mud = get_GSM_proportion(classes_φ, frequency)
    if gravel > 0.0:
        return get_GSM_textural_group_BP12(gravel, sand, mud)
    else:
        sand, silt, clay = get_SSC_proportion(classes_φ, frequency)
        return get_SSC_textural_group_BP12(sand, silt, clay)

def get_moments(classes_μm: np.ndarray, classes_φ: np.ndarray, frequency: np.ndarray, FW57=False):
    reverse_phi_ppf = get_ppf(classes_φ, frequency)
    if FW57:
        geometric_moments = geometric_FW57(reverse_phi_ppf)
        logarithmic_moments = logarithmic_FW57(reverse_phi_ppf)
    else:
        geometric_moments = geometric(classes_μm, frequency)
        logarithmic_moments = logarithmic(classes_φ, frequency)
    mean_φ = logarithmic_moments["mean"]
    mean_description = string.capwords(" ".join(scale_description(mean_φ)))
    geometric_moments["mean_description"] = mean_description
    logarithmic_moments["mean_description"] = mean_description

    max_pos = np.unravel_index(np.argmax(frequency), frequency.shape)
    geometric_moments["mode"] = classes_μm[max_pos]
    geometric_moments["modes"] = get_modes(classes_μm, classes_φ, frequency, geometric=True)[0]
    logarithmic_moments["mode"] = classes_φ[max_pos]
    logarithmic_moments["modes"] = get_modes(classes_μm, classes_φ, frequency, geometric=False)[0]

    median_φ = reverse_phi_ppf(0.5).max()
    geometric_moments["median"] = convert_φ_to_μm(median_φ)
    logarithmic_moments["median"] = median_φ

    GSM_proportion = get_GSM_proportion(classes_φ, frequency)
    SSC_proportion = get_SSC_proportion(classes_φ, frequency)
    BGSSC_proportion = get_BGSSC_proportion(classes_φ, frequency)
    geometric_moments["GSM_proportion"] = GSM_proportion
    geometric_moments["SSC_proportion"] = SSC_proportion
    geometric_moments["BGSSC_proportion"] = BGSSC_proportion
    logarithmic_moments["GSM_proportion"] = GSM_proportion
    logarithmic_moments["SSC_proportion"] = SSC_proportion
    logarithmic_moments["BGSSC_proportion"] = BGSSC_proportion

    Folk54_description = get_textural_group_Folk54(classes_φ, frequency)
    geometric_moments["textural_group_Folk54"] = Folk54_description
    logarithmic_moments["textural_group_Folk54"] = Folk54_description

    BP12_symbol_strs, BP12_description_strs = get_textural_group_BP12(classes_φ, frequency)
    BP12_symbol = "".join(BP12_symbol_strs)
    BP12_description = string.capwords(" ".join(BP12_description_strs))
    geometric_moments["textural_group_BP12_symbol"] = BP12_symbol
    geometric_moments["textural_group_BP12"] = BP12_description
    logarithmic_moments["textural_group_BP12_symbol"] = BP12_symbol
    logarithmic_moments["textural_group_BP12"] = BP12_description

    return geometric_moments, logarithmic_moments
