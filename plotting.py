import matplotlib.animation as animation
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import utils as UT


def plot_pixel_spectra(data, wavelengths, x_coords, y_coords, show_plot=False, plot_separately=False):

    # If both x and y coordinates point to just one pixel, only plot that pixel
    if type(x_coords) == int and type(y_coords) == int:
        spectrum = data[y_coords, x_coords, :]
        fig = plt.figure()
        plt.plot(wavelengths, spectrum)
        plt.ylabel('Radiance [W / (m² sr nm)]')
        plt.xlabel('Wavelength [nm]')
        # Plot vertical lines at some expected spike locations
        plt.vlines(x=(1140, 2208), ymin=-0.0001, ymax=0.01, colors='r', linestyles=':')
        plt.annotate(text='1140 nm', xy=(1120, 0.0101), color='r')
        plt.annotate(text='2208 nm', xy=(2188, 0.0101), color='r')
        plt.savefig('figs/spectrum.png')
        if show_plot:
            plt.show()
        plt.close(fig)
        return spectrum

    spectra = []
    for i in range(y_coords[0], y_coords[1] + 1):
        for j in range(x_coords[0], x_coords[1] + 1):
            spectra.append(data[i, j, :])
    mean_spectrum = np.mean(np.asarray(spectra), axis=0)

    if plot_separately:
        fig = plt.figure()
        plt.plot(wavelengths, mean_spectrum)
        plt.ylabel('Radiance [W / (m² sr nm)]')
        plt.xlabel('Wavelength [nm]')

        plt.figure()
        for spectrum in spectra:
            plt.plot(wavelengths, spectrum)
        plt.ylabel('Radiance [W / (m² sr nm)]')
        plt.xlabel('Wavelength [nm]')
        plt.savefig('figs/individual_spectral.png')

        plt.show()

    else:
        fig, axs = plt.subplots(1, 2)
        for spectrum in spectra:
            axs[0].plot(wavelengths, spectrum)
        axs[0].set_title('individual pixel spectra')

        axs[1].plot(wavelengths, mean_spectrum)
        axs[1].set_title('mean spectrum')
    if show_plot:
        plt.show()
    plt.close(fig)

    return mean_spectrum


def plot_rgb_reconstruction(data, wavelengths, RGB_center_wavelengths = (640, 540, 460), averaged_channel_count=10, show_plot=False):
    R_center, G_center, B_center = RGB_center_wavelengths[0], RGB_center_wavelengths[1], RGB_center_wavelengths[2]  #2208, 1140, 817

    R_center_index = UT.find_nearest(wavelengths, R_center)
    G_center_index = UT.find_nearest(wavelengths, G_center)
    B_center_index = UT.find_nearest(wavelengths, B_center)

    if averaged_channel_count > 1:
        R_data = data[:, :, int(R_center_index - (averaged_channel_count/2)):int(R_center_index + (averaged_channel_count/2))]
        R_data = np.mean(R_data, axis=2)
        G_data = data[:, :, int(G_center_index - (averaged_channel_count / 2)):int(G_center_index + (averaged_channel_count / 2))]
        G_data = np.mean(G_data, axis=2)
        B_data = data[:, :, int(B_center_index - (averaged_channel_count / 2)):int(B_center_index + (averaged_channel_count / 2))]
        B_data = np.mean(B_data, axis=2)
    else:
        R_data = data[:, :, R_center_index]
        R_data = R_data / np.max(R_data)
        G_data = data[:, :, G_center_index]
        G_data = G_data / np.max(G_data)
        B_data = data[:, :, B_center_index]
        B_data = B_data / np.max(B_data)
    RGB_data = np.dstack([R_data, G_data, B_data])
    RGB_data = np.nan_to_num(RGB_data ** 0.5, nan=0)  # gamma
    RGB_data = RGB_data / (np.max(RGB_data) * 0.1)

    fig = plt.figure()
    plt.imshow(RGB_data)
    # plt.title('RGB reconstruction')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('./figs/RGB_reconstruction.png')
    if show_plot:
        plt.show()
    plt.close(fig)

    return RGB_data


def plot_integrated_radiance(data, show_plot=False, log_y=True):
    # Plotting radiance integrated over wavelengths
    plottable_sum = np.sum(data, axis=2)
    fig = plt.figure()
    if log_y:
        plt.imshow(plottable_sum, norm=matplotlib.colors.LogNorm())#, vmin=0, vmax=5)
    else:
        plt.imshow(plottable_sum)
    # plt.colorbar()
    # plt.title('Integrated radiance')
    plt.savefig('./figs/integrated_radiance.png')
    if show_plot:
        plt.show()
    plt.close(fig)
    return plottable_sum


def animated_subplot(cube, wavelengths, title=None):

    ims = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    for imgNum in range(len(wavelengths)):
        img = cube[:, :, imgNum]
        frame = ax.imshow(img)
        if imgNum < 91:
            t = ax.annotate(f'VNIR, wavelength: {wavelengths[imgNum]:.2f} nm', (1, 1), color='yellow')  # add text
        else:
            t = ax.annotate(f'SWIR, wavelength: {wavelengths[imgNum]:.2f} nm', (1, 1), color='yellow')  # add text
        ims.append([frame, t])  # add both the image and the text to the list of artists
    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=500)
    # anim.savefig(f'./{title}.gif')
    anim.save(f'./figs/{title}.gif')
    plt.show()

