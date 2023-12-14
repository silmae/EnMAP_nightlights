import numpy as np
from matplotlib import pyplot as plt
import scipy


def toy_data(samples: int, lines: int,  channels: int, signal):
    """Build fake spectral image with dimensions (samples, lines, channels), where samples denotes number of
    across-track spatial pixels, channels denotes number of spectral channels, and lines denotes number of
    different exposures."""

    # Set standard deviation of noise as multiple of signal maximum
    noise_scale = max(signal) * 3

    # The matched filter in spectral wants the spectral dimension to be the last one, which dictates the dimension order
    data = np.zeros((samples, lines, channels))

    # Add noise to each pixel
    for i in range(samples):
        for j in range(lines):
            data[i, j, :] = np.random.normal(0, noise_scale, channels)

    # Add signal on top of noise for some random pixels and ones below them, creating vertical streaks
    np.random.seed(42)
    star_x_indices = np.random.randint(low=0, high=lines, size=20)
    star_y_indices = np.random.randint(low=0, high=samples - 10, size=20)
    star_indices = tuple(zip(star_y_indices, star_x_indices))
    for i in range(samples):
        for j in range(lines):
            if (i, j) in star_indices: #or i == j:
                for n in range(10):
                    data[i+n, j, :] = data[i+n, j, :] + signal

    # Plot a spectrum with signal and a spectrum without signal into same figure
    plt.figure()
    plt.plot(np.random.normal(0, noise_scale, channels) + signal, label='with signal')
    plt.plot(np.random.normal(0, noise_scale, channels), label='without signal')
    plt.legend()
    plt.show()
    return data


def thermal_radiance(T: float, emissivity: float or list or np.ndarray, wavelengths: np.ndarray, wl_unit: 'um'):
    """
    Function taken from https://github.com/silmae/AsTherCorNN/blob/main/radiance_data.py.

    Calculate and return approximate thermally emitted spectral radiance using Planck's law. Angle
    dependence of emitted radiance is approximated as Lambertian.

    :param T:
        Surface temperature, in Kelvins
    :param emissivity:
        Emissivity = sample emission spectrum divided by ideal blackbody spectrum of same temperature. Float or
        vector/list with same number of elements as reflectance.
    :param wavelengths:
        Wavelengths where the emission is to be calculated
    :param wl_unit:
        Unit of wavelengths: "um" for micrometer or "nm" for nanometer

    :return L_th:
        Spectral radiance emitted by the surface.
    """

    # Physical constants, fetched from scipy library
    c = scipy.constants.c  # 2.998e8 m / s, speed of light in vacuum
    kB = scipy.constants.Boltzmann  # 1.381e-23 m² kg / s² / K (= J / K), Boltzmann constant
    h = scipy.constants.h  # 6.626e-34 m² kg / s (= J s), Planck constant

    if type(emissivity) == float or type(emissivity) == np.float64 or type(emissivity) == np.float32 or len(emissivity) == 1:
        # If a single float, make it into a vector where each element is that number
        eps = np.empty((len(wavelengths), 1))
        eps.fill(emissivity)
    elif type(emissivity) == list:
        # If emissivity is a list with correct length, convert to ndarray
        if len(emissivity) == len(wavelengths):
            eps = np.asarray(emissivity)
        else:
            print('Emissivity list was not same length as wavelength vector. Stopping execution...')
            quit()
    elif type(emissivity) == np.ndarray:
        # If emissivity array is of correct shape, rename it to emittance and proceed
        if emissivity.shape == wavelengths.shape or emissivity.shape == (
        wavelengths.shape, 1) or emissivity.shape == (1, wavelengths.shape):
            eps = emissivity
        else:
            print('Emissivity array was not same shape as wavelength vector. Stopping execution...')
            quit()

    if wl_unit == 'um':
        pass
    elif wl_unit == 'nm':
        wavelengths = wavelengths / 1000  # convert from nm to um

    L_th = np.zeros((len(wavelengths), 2))
    L_th[:, 0] = wavelengths

    for i in range(len(wavelengths)):
        wl = wavelengths[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i, 1] = eps[i] * (2 * h * c ** 2) / ((wl ** 5) * (np.exp((h * c) / (wl * kB * T)) - 1))  # Planck's law
        L_th[i, 1] = L_th[i, 1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    if wl_unit == 'um':
        pass
    elif wl_unit == 'nm':
        L_th[:, 1] = L_th[:, 1] / 1000  # convert from 1/µm to 1/nm
        L_th[:, 0] = L_th[:, 0] * 1000  # convert from µm to nm

    # plt.figure()
    # plt.plot(L_th[:, 0], L_th[:, 1])
    # plt.show()

    return L_th


def generate_bb_spectra(temperature_min, temperature_max, temperature_spacing, wavelengths):
    """Generate a series of blackbody spectra with temperatures given as argument, using the supplied
    wavelength vector (in micrometers) an emissivity of 1. Returns a list of dictionaries, where each dictionary
    contains a 'description' (the temperature) and a 'spectrum' (thermally emitted spectral radiance. """

    # Create vector of temperatures
    temperatures = range(temperature_min, temperature_max, temperature_spacing)
    # List for the generated spectra
    generated = []
    for temperature in temperatures:
        # Thermally emitted spectral radiance
        spectral_radiance = thermal_radiance(T=temperature, emissivity=1.0, wavelengths=wavelengths, wl_unit='nm')[:, 1]
        # Store spectrum and temperature in dictionary and append it to a list
        result_dict = {'description': f'{temperature} K',
                       'spectrum': spectral_radiance}
        generated.append(result_dict)

    return generated
