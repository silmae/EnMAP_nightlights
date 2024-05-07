
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['figure.constrained_layout.use'] = True

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 200})
plt.rcParams.update({'savefig.bbox': 'tight'})
# LaTeX font for all text in all figures
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

import matplotlib
import scipy
from PIL import Image
import cv2
import skimage
import spectral

import file_handling as FH
import data_generation as DG
import analysis as AN
import utils as UT
import plotting as PT



if __name__ == '__main__':

    ####### Sandbox #######
    # FH.open_envi_image('./data/EnMAP/L1C_cropped_co-registered/riad/riad_clipped.hdr', './data/EnMAP/L1C_cropped_co-registered/riad/riad_clipped.')

    # 2 and 3 are almost perfectly co-registered
    # # 8.8.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_1', type='vnir')
    # 12.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_2')
    # # 8.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_3')
    # # 16.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_5')  # the radiance values are ten times higher on this one?

    # RGB_image = PT.plot_rgb_reconstruction(data, wavelengths, show_plot=True)

    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_6')  # nothing in frame? cloudy?
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_7')  # nothing in frame?
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_4') # not on target, noisy

    # Luxor: indices_y=(362, 371), indices_x=(212, 221), other light: indices_y=(344, 351), indices_x=(439, 446)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='vnir')  # 8.8.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_2', type='both')  # 12.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_3', type='both')  # 8.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')  # 16.7.2023
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5_corrected', type='both')  # 16.7.2023

    ###### Checking co-registration of EnMAP VNIR and SWIR with plots ######
    # _, data_vnir, wavelengths_vnir = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='vnir', indices_y=(362, 371), indices_x=(212, 221))
    # _, data_swir, wavelengths_swir = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='swir', indices_y=(362, 371), indices_x=(212, 221))
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.mean(data_vnir, axis=2))
    # axs[0].set_title('VNIR')
    # axs[1].imshow(np.mean(data_swir, axis=2))
    # axs[1].set_title('SWIR')
    # plt.savefig('./figs/spot_comparison.png')
    # plt.show()

    # PT.animated_subplot(cube=data, wavelengths=wavelengths, title='Luxor Sky Beam on EnMAP')

    # Las Vegas
    envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')
    # Riad
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C_other_cities/Riad/dims_op_oc_oc-en_701066037_3')  # 2 and 3 are great! The rest not so much
    # Tokyo:
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C_other_cities/Tokyo/dims_op_oc_oc-en_701066090_1')  # 1, 2, 3, 4, 5, (6) (7, 8, 9 show next to no lights)
    # Las Vegas daytime image
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C/dims_op_oc_oc-en_701052184_3',
    #                                                            indices_y=(800, 1100), indices_x=(500, 700))

    # # # Cropped images from Tokyo
    # envi_img, data1, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/tokyo/Tokyo_4')
    # envi_img, data2, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi(
    #     './data/EnMAP/L1C_cropped_co-registered/tokyo/Tokyo_6')
    #
    # data1 = (np.mean(data1, axis=2) ** 0.5) * 100
    # data2 = (np.mean(data2, axis=2) ** 0.5) * 100
    # data3 = np.zeros(shape=data1.shape)
    #
    # data = np.dstack([data1, data2, data3])
    #
    # plt.figure()
    # plt.imshow(data)
    # plt.show()

    # RGB_image = PT.plot_rgb_reconstruction(data, wavelengths, show_plot=True)
    # integrated_radiance = PT.plot_integrated_radiance(data, show_plot=True, log_y=True)
    #
    # AN.calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=True)
    # AN.calculate_spectral_G_index(data, wavelengths, show_plot=True)

    ######## Calculating luminous efficiency of radiation (LER) ########

    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_3', type='both')  # 8.7.2023
    # efficiency1 = AN.calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=True)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_2', type='both')  # 12.7.2023
    # efficiency2 = AN.calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=False)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')  # 16.7.2023
    # efficiency3 = AN.calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=False)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='both')  # 8.8.2023
    # efficiency4 = AN.calculate_luminous_efficiency_of_radiation(data, wavelengths, show_plot=False)
    #
    # plt.figure()
    # plt.imshow(efficiency3 - efficiency2)
    # plt.show()

    ######## Calculating spectral G index ########

    # # Larger G_n value means newer data: compare by subtracting older from newer
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_3', type='both')  # 8.7.2023
    # G_1 = AN.calculate_spectral_G_index(data, wavelengths, show_plot=False)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_2', type='both')  # 12.7.2023
    # G_2 = AN.calculate_spectral_G_index(data, wavelengths, show_plot=False)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')  # 16.7.2023
    # G_3 = AN.calculate_spectral_G_index(data, wavelengths, show_plot=False)
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='both')  # 8.8.2023
    # G_4 = AN.calculate_spectral_G_index(data, wavelengths, show_plot=False)
    #
    # plt.figure()
    # plt.imshow(G_3 - G_2)
    # plt.show()

    # im = Image.fromarray(G_3 - G_2)
    # im.save(f"./figs/G_index_difference.TIFF")

    ######## Matched filtering the loaded data and plotting results ########

    # Las Vegas
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='both')
    # Riad
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C_other_cities/Riad/dims_op_oc_oc-en_701066037_2')  # 2 and 3 are great! The rest not so much
    # Tokyo:
    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C_other_cities/Tokyo/dims_op_oc_oc-en_701066090_2')  # 1, 2, 3, 4, 5, (6) (7, 8, 9 show next to no lights)

    # Load target spectra: light sources
    targets = FH.load_spectral_library(wavelengths, MH=False, LED=True, HPS=False, LPS=False, MV=False, FL=False)

    # Resample the image cube and target signals to RGB
    RGB_wavelengths = [450, 550, 650]
    RGB_FWHMs = [50, 50, 50]

    resample = spectral.BandResampler(wavelengths, RGB_wavelengths, FWHMs, RGB_FWHMs)

    data_resampled = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i, j, :] = resample(data[i, j, :])
    data = data_resampled

    targets_resampled = np.copy(targets)
    for i, target in enumerate(targets):
        spectrum = resample(target['spectrum'])
        targets_resampled[i]['spectrum'] = spectrum
    targets = targets_resampled
    print('')

    # plt.figure()
    # plt.plot(wavelengths[:60], targets[4]['spectrum'][:60])
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Intensity [arbitrary units]')
    # plt.show()

    # # # # Generate targets: blackbody spectra with different temperatures
    # # targets = DG.generate_bb_spectra(temperature_min=3000, temperature_max=10000, temperature_spacing=500, wavelengths=wavelengths)
    # # sun_target = {'description': 'Sun irradiance',
    # #                'spectrum': FH.load_and_resample_solar_spectrum('./data/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc', wavelengths)}
    # # targets.append(sun_target)
    #
    # Run matched filters on the data
    filtered_images = AN.run_matched_filtering(data, targets, show_plots=True, save_tiff=False)
    #
    # # Plot map highlighting sharp emission spikes at certain wavelengths
    # AN.plot_one_channel_spike_map(data, wavelengths, spike_wl1=2208, spike_wl2=1140, binary_map=True)  # 2208, 1140, 817
    # Plot spectrum of an area that has both spikes
    # spectrum = PT.plot_pixel_spectra(data, wavelengths, y_coords=303, x_coords=232, show_plot=True)

    # # Plot spectra of certain area
    # PT.plot_pixel_spectra(data, wavelengths, y_coords=(366, 368), x_coords=(215, 219), show_plot=True)
    # PT.plot_pixel_spectra(data, wavelengths, y_coords=(697, 699), x_coords=(608, 611))
    # # The Sphere
    # PT.plot_pixel_spectra(data[:, :, :50], wavelengths[:50], y_coords=(269, 272), x_coords=(253, 257), show_plot=True)
    # Smaller area of the sphere
    # PT.plot_pixel_spectra(data[:, :, :50], wavelengths[:50], y_coords=(270, 271), x_coords=(254, 256), show_plot=True, plot_separately=True)


    ########### Plots to show differences in spectra from different nights ###########

    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')
    # mean_luxor_5 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(365, 367), x_coords=(215, 217))
    # MGM_5 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(342, 344), x_coords=(233, 236))
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_5')
    # luxor_offnadir_angle5 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[738, 534])
    # MGM_offnadir_angle5 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[715, 553])
    # # PT.plot_integrated_radiance(data, show_plot=True)
    #
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi(
    #     'data/EnMAP/L1C_cropped_co-registered/LasVegas_1', type='both')
    # mean_luxor_1 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(366, 368), x_coords=(216, 218))
    # MGM_1 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(342, 344), x_coords=(233, 236))
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_1')
    # luxor_offnadir_angle1 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[698, 610])
    # MGM_offnadir_angle1 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[675, 628])
    # # PT.plot_integrated_radiance(data, show_plot=True)
    #
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi(
    #     'data/EnMAP/L1C_cropped_co-registered/LasVegas_2', type='both')
    # mean_luxor_2 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(365, 367), x_coords=(215, 217))
    # MGM_2 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(342, 344), x_coords=(233, 236))
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_2')
    # luxor_offnadir_angle2 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[920, 556])
    # MGM_offnadir_angle2 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[897, 574])
    # # PT.plot_integrated_radiance(data, show_plot=True)
    #
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi(
    #     'data/EnMAP/L1C_cropped_co-registered/LasVegas_3', type='both')
    # mean_luxor_3 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(366, 368), x_coords=(215, 217))
    # MGM_3 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(343, 345), x_coords=(233, 236))
    # envi_img, data, wavelengths, FWHMs = FH.load_EnMAP_data_with_envi('data/EnMAP/L1C/dims_op_oc_oc-en_701045151_3')
    # luxor_offnadir_angle3 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[905, 549])
    # MGM_offnadir_angle3 = UT.interpolate_offNadir_angle(data, envi_img, coordinates=[883, 567])
    # # PT.plot_integrated_radiance(data, show_plot=True)
    #
    # plt.figure()
    # plt.plot(wavelengths, mean_luxor_1 , label=f'Off-nadir angle {luxor_offnadir_angle1:.2f}$^\circ$')
    # plt.plot(wavelengths, mean_luxor_2 , label=f'Off-nadir angle {luxor_offnadir_angle2:.2f}$^\circ$', linestyle='--')
    # plt.plot(wavelengths, mean_luxor_3 , label=f'Off-nadir angle {luxor_offnadir_angle3:.2f}$^\circ$', linestyle=':')
    # plt.plot(wavelengths, mean_luxor_5 , label=f'Off-nadir angle {luxor_offnadir_angle5:.2f}$^\circ$', linestyle='-.')
    # plt.ylabel('Radiance [W / (m² sr nm)]')
    # plt.xlabel('Wavelength [nm]')
    # plt.legend()
    # plt.savefig('figs/luxor_angle_dependence.png')
    #
    # plt.figure()
    # plt.plot(wavelengths[:50], MGM_1[:50], label=f'Off-nadir angle {MGM_offnadir_angle1:.2f}$^\circ$')
    # plt.plot(wavelengths[:50], MGM_2[:50], label=f'Off-nadir angle {MGM_offnadir_angle2:.2f}$^\circ$', linestyle='--')
    # plt.plot(wavelengths[:50], MGM_3[:50], label=f'Off-nadir angle {MGM_offnadir_angle3:.2f}$^\circ$', linestyle=':')
    # plt.plot(wavelengths[:50], MGM_5[:50], label=f'Off-nadir angle {MGM_offnadir_angle5:.2f}$^\circ$', linestyle='-.')
    # plt.ylabel('Radiance [W / (m² sr nm)]')
    # plt.xlabel('Wavelength [nm]')
    # plt.legend()
    # plt.savefig('figs/MGM_angle_dependence.png')
    #
    # plt.show()

    # envi_img, data, wavelengths = FH.load_EnMAP_data_with_envi('./data/EnMAP/L1C_cropped_co-registered/LasVegas_5', type='both')
    # mean_luxor_5 = PT.plot_pixel_spectra(data, wavelengths, y_coords=(366, 367), x_coords=(216, 217), show_plot=True)
    #
    # PT.plot_pixel_spectra(data, y_coords=38, x_coords=182)
    # PT.plot_pixel_spectra(data, y_coords=200, x_coords=800)
    # # Another xenon lamp?
    # PT.plot_pixel_spectra(data, wavelengths, y_coords=347, x_coords=442)












    ########## Likely trash, saved for snippets ##########

    # # Plot RGB image from filter activations of red, green, and blue LEDs
    # r = np.zeros(data[:, :, 0].shape)
    # g = np.zeros(data[:, :, 0].shape)
    # b = np.zeros(data[:, :, 0].shape)
    #
    # for i, target in enumerate(targets):
    #     description = target['description']
    #     if 'Red' in description:
    #         r = r + filtered_images[i]
    #     elif 'Green' in description:
    #         g = g + filtered_images[i]
    #     elif 'Blue' in description:
    #         b = b + filtered_images[i]
    #
    # RGB = np.zeros(data[:, :, :3].shape)
    # RGB[:, :, 0] = abs(r) / np.max(abs(r))
    # RGB[:, :, 1] = abs(g) / np.max(abs(g))
    # RGB[:, :, 2] = abs(b) / np.max(abs(b))
    #
    # plt.figure()
    # plt.imshow(RGB ** (1/2))
    # plt.title('R, G, and B filter activations')
    # plt.savefig(f'./figs/RGB_LED.png')
    # plt.show()


    ######## Opening EnMAP image using the EnPT package ########  [usually better to just use the ENVI hdr in the folder]

    # # # The Las Vegas image used in article https://doi.org/10.3390/rs15164025
    # data, wavelengths = FH.load_EnMAP_data(
    # path='/home/leevi/PycharmProjects/starfinder/data/EnMAP/dims_op_oc_oc-en_701027473_2/ENMAP.HSI.L1B/ENMAP-HSI-L1BDT0000004966_02-2022-11-03T06:10:55.404_julejoli-cat1distributor_701027471_722630258_2023-11-14T20:16:41.727/ENMAP01-____L1B-DT0000004966_20221103T061055Z_002_V010400_20231114T115959Z.ZIP',
    #     type='vnir',
    #     indices_x=(450, 650),
    #     indices_y=(450, 750)
    # )

    # data, wavelengths = FH.load_EnMAP_L1B_data_with_EnPT(folder_path='./data/EnMAP/L1B/dims_op_oc_oc-en_701021709_2')


    # # Trying RX anomaly detection
    # filtered = detectors.rx(data)
    # from scipy.stats import chi2
    # nbands = data.shape[-1]
    # P = chi2.ppf(1-7e-17, nbands)
    # filtered = (1 * (filtered > P))

    # # Trying euclidean distance instead of matched filter
    # distances = np.zeros((len(data[:, 0, 0]), len(data[0, :, 0])))
    # for i in range(len(data[:, 0, 0])):
    #     for j in range(len(data[0, :, 0])):
    #         distances[i, j] = scipy.spatial.distance.euclidean(data[i, j], spectrum)
    # filtered = 1 - distances

    # # Denoising filter response
    # filtered = skimage.restoration.denoise_tv_chambolle(image=filtered)

    # # Plot all filter outputs as subplots of one figure
    # plt.figure()
    # # set number of columns
    # ncols = 6
    # # calculate number of rows
    # nrows = len(filtered_images) // ncols + (len(filtered_images) % ncols > 0)
    #
    # # loop through the length of image list and keep track of index
    # for n, filtered in enumerate(filtered_images):
    #     # add a new subplot iteratively using nrows and cols
    #     ax = plt.subplot(nrows, ncols, n + 1)
    #
    #     ax.imshow(filtered)
    #     ax.set_title(targets[n]['description'])
    #
    # plt.tight_layout()
    # plt.show()

    # # Plot sum of filter output images
    # plt.figure()
    # plt.imshow(sum(filtered_images))
    # plt.show()

    # # Create signal to be detected
    # filter_signal = thermal_radiance(6000, 1.0, wavelengths, wl_unit='nm')[:, 1]
    # # Apply the filter
    # filtered = detectors.matched_filter(data, filter_signal)
    #
    # fig, axs = plt.subplots(2,2)
    #
    # ax = axs[0,0]
    # ax.imshow(data[:,:,20])
    # ax.set_title('original, one channel')
    #
    # ax = axs[0,1]
    # ax.imshow(filtered)
    # ax.set_title('filter output')
    #
    # ax = axs[1,0]
    # ax.plot(data[0, 0, :])
    # ax.plot(data[1, 1, :])
    # ax.plot(data[2, 2, :])
    # ax.set_title('spectra with signal')
    #
    # ax = axs[1, 1]
    # ax.plot(data[1, 0, :])
    # ax.plot(data[2, 0, :])
    # ax.plot(data[3, 0, :])
    # ax.set_title('spectra without signal')
    #
    # plt.tight_layout()
    # plt.show()
