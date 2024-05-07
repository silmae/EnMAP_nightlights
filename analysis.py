import scipy
from spectral.algorithms import detectors
from PIL import Image
import cv2
import skimage
import numpy as np
import spectral
from matplotlib import pyplot as plt
import matplotlib

import file_handling as FH
import utils as UT


def plot_one_channel_spike_map(data, wavelengths, spike_wl1, spike_wl2, binary_map=True):
    def one_channel_spike_map(data, wavelengths, spike_wl, binary_map=True, min_threshold=None):
        spike_index = UT.find_nearest(spike_wl, wavelengths)
        spike_background = np.dstack([data[:, :, spike_index - 1], data[:, :, spike_index + 1]])
        spike_background = np.mean(spike_background, axis=2)
        spike_map = abs(data[:, :, spike_index] - spike_background)

        spike_map = spike_map / np.max(spike_map)
        if binary_map:
            # min_threshold = 0.035
            spike_map = np.clip(spike_map, a_min=min_threshold, a_max=1) - min_threshold
            spike_map = np.clip(spike_map * 1000, a_min=0, a_max=1)

        # fig = plt.figure()
        # plt.imshow(spike_map)
        # plt.show()
        # plt.close(fig)
        return spike_map
    spike_map1 = one_channel_spike_map(data, wavelengths, spike_wl=spike_wl1, binary_map=binary_map, min_threshold=0.035)
    spike_map2 = one_channel_spike_map(data, wavelengths, spike_wl=spike_wl2, binary_map=binary_map, min_threshold=0.040)
    radiance_sum = plottable_sum = np.sum(data, axis=2)
    spike_RGB_plottable = np.dstack([radiance_sum, radiance_sum, radiance_sum])
    for i in range(3):
        spike_RGB_plottable[:, :, i] = radiance_sum * 5 / np.max(plottable_sum) - spike_map1 - spike_map2

    # spike_RGB_plottable = spike_RGB_plottable ** 0.5 - 0.1
    spike_RGB_plottable[:, :, 0] = spike_RGB_plottable[:, :, 0] + (spike_map1 * 3)  # Red
    spike_RGB_plottable[:, :, 1] = spike_RGB_plottable[:, :, 1] + (spike_map2 * 3)  # Green

    spike_RGB_plottable = spike_RGB_plottable ** 0.5

    # plot_pixel_spectra(data, wavelengths, y_coords=304, x_coords=231, show_plot=True)

    fig = plt.figure()
    plt.imshow(spike_RGB_plottable)
    # plt.title(f'Map of sharp emission spikes at {spike_wl1} nm (red) and {spike_wl2} nm (green)')
    plt.savefig('figs/atomic_spike_map.png')
    plt.show()
    plt.close(fig)


def run_matched_filtering(data, targets, show_plots=False, save_tiff=False):
    filtered_images = []

    for target in targets:
        description = target['description']
        spectrum = target['spectrum']

        # spectrum = spectrum / np.max(spectrum)  # normalizing the spectrum to hopefully get values closer to the target's radiance

        # Calculate background statistics from a subset of data "blended" with zeros. For some reason this works, adjust the mount of zeros as needed
        # background_data = np.zeros((100, 190, len(wavelengths)))
        # background_data[:100, :100, :] = data[:100, :100, :]
        # background_data = data[:50, :50, :]
        # background = spectral.calc_stats(background_data)

        # Apply matched filter
        filtered = detectors.matched_filter(data, spectrum)#, background=background)
        threshold = 0.0001  # do not allow negative correlation
        filtered = np.clip(filtered, a_min=threshold, a_max=1000)# - threshold  # thresholding
        filtered_images.append(filtered)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(filtered, norm=matplotlib.colors.LogNorm())  # , vmin=0.004, vmax=0.01)
        ax.set_title(f'filter output: {description}')
        plt.savefig(f'./figs/{description}.png')
        if show_plots is True:
            plt.show()
        if save_tiff is True:
            im = Image.fromarray(filtered)
            im.save(f"./figs/{description}.TIFF")
        plt.close(fig)

        # plt.figure()
        # plt.plot(spectrum)
        #
        # plt.figure()
        # plt.plot(data[366, 222, :])
        # plt.title(f'{filtered[366, 222]}')

        # plt.show()

    return filtered_images


def calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=True, reference_plot=False):
    Vlambda = FH.load_and_resample_Vlambda('data/Vlambda_1nm.csv', wavelengths)
    human_weighed = np.sum(data * Vlambda, axis=2)
    integrated_radiance = np.sum(data, axis=2)
    efficiency = human_weighed / integrated_radiance

    if reference_plot:
        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        ax.imshow(integrated_radiance, norm=matplotlib.colors.LogNorm())
        ax.set_title('Integrated radiance')

        ax = axs[1]
        ax.imshow(efficiency, vmin=0, vmax=0.5)
        ax.set_title('Luminous efficiency of radiation')
    else:
        fig = plt.figure()
        plt.imshow(efficiency)

    plt.savefig('./figs/LER.png')

    if show_plot:
        plt.show()
    plt.close(fig)

    return efficiency


def calculate_spectral_G_index(data, wavelengths, show_plot=True):
    blue_cutoff_index = UT.find_nearest(wavelengths, 500)
    blue = np.sum(data[:, :, :blue_cutoff_index], axis=2)

    Vlambda = FH.load_and_resample_Vlambda('data/Vlambda_1nm.csv', wavelengths)
    visual = np.sum(data * Vlambda, axis=2)

    G = -2.5 * np.log10(blue / visual)

    # G = cv2.medianBlur(G.astype('float32'), ksize=3)

    fig = plt.figure()
    plt.imshow(G)
    plt.savefig('./figs/spectral_G_index.png')
    if show_plot:
        plt.show()
    plt.close(fig)

    return G

