import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

invalid_moments = dict(mean=np.nan, std=np.nan, skewness=np.nan, kurtosis=np.nan)

def get_modes(classes_μm, classes_φ, frequency, geometric=True):
    peaks, _ = find_peaks(frequency)
    position = classes_μm[peaks] if geometric else classes_φ[peaks]
    values = frequency[peaks]
    return tuple(position), tuple(values)

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

def mean_description(φ):
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
    else:
        return "Clay"

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

def get_moments(classes_μm: np.ndarray, classes_φ: np.ndarray, frequency: np.ndarray, FW57=False):
    reverse_phi_ppf = get_ppf(classes_φ, frequency)
    if FW57:
        geometric_moments = geometric_FW57(reverse_phi_ppf)
        logarithmic_moments = logarithmic_FW57(reverse_phi_ppf)
    else:
        geometric_moments = geometric(classes_μm, frequency)
        logarithmic_moments = logarithmic(classes_φ, frequency)
    mean_φ = logarithmic_moments["mean"]
    geometric_moments["mean_description"] = " ".join(mean_description(mean_φ))
    logarithmic_moments["mean_description"] = " ".join(mean_description(mean_φ))

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

    return geometric_moments, logarithmic_moments

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from scipy.stats import norm
    plt.style.use(["science", "no-latex"])

    classes_μm = np.logspace(0, 5, 101) * 0.02
    classes_φ = convert_μm_to_φ(classes_μm)

    distribution = 0.2*norm.pdf(classes_φ, 8, 1) + 0.8*norm.pdf(classes_φ, 4, 1)
    distribution /= np.sum(distribution)
    distribution = np.round(distribution, 4)
    start = time.time()
    for i in range(10000):
        geometric_moments, logarithmic_moments = get_moments(classes_μm, classes_φ, distribution, FW57=False)
    end = time.time()
    print(f"calculate 10000 times, spent {end-start} s")
