import csv
import glob
import math
import os
from pathlib import Path

from enpt.execution.controller import EnPT_Controller
import spectral.io.envi as envi
import spectral
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from lxml import objectify #etree
import lxml

import utils


# import xml.etree.ElementTree as etree

def load_csv(filepath, delimiter='\t'):
    """Load csv from file into ndarray"""

    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = []
        for row in reader:
            data.append(row)
    csvfile.close()
    return np.asarray(data)


def load_and_resample_solar_spectrum(path, wavelengths):
    ds = nc.Dataset(path)
    original_wls = np.asarray(ds.variables['Vacuum Wavelength'][:])
    irradiance = np.asarray(ds.variables['SSI'][:])
    integrated = sum(irradiance)
    resample = spectral.BandResampler(original_wls, wavelengths)
    irradiance = resample(irradiance)

    # Convert irradiance to radiance: divide by the solid angle of the Sun as seen from Earth
    radiance = irradiance / 6.794e-5

    # plt.figure()
    # plt.plot(wavelengths, radiance)
    # plt.show()

    return radiance


def load_and_resample_Vlambda(path, wavelengths, plot=False):
    """Loading and resampling the spectral sensitivity of human photopic (daytime) vision: the V lambda curve."""
    data = load_csv(path)
    spectrum = data[:, 1].astype(float)
    orig_wavelengths = data[:, 0].astype(float)
    resample = spectral.BandResampler(orig_wavelengths, wavelengths)
    resampled_spectrum = resample(spectrum)
    resampled_spectrum = np.nan_to_num(resampled_spectrum, nan=0)  # Values missing in original spectrum will be NaNs

    if plot:
        plt.figure()
        plt.plot(wavelengths[:55], resampled_spectrum[:55])
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('$V_\lambda$')
        plt.show()

    return resampled_spectrum


def load_spectral_library(wavelengths, MH=True, LED=True, HPS=True, LPS=True, MV=True, FL=True):
    # Load target spectra: light sources
    targets = []
    if MH:
        light_source_spectra_MH = load_and_resample_light_spectra('./data/light_spectra/metal_halide.csv', wavelengths)
        targets = targets + light_source_spectra_MH
    if LED:
        light_source_spectra_LED = load_and_resample_light_spectra('./data/light_spectra/LED.csv', wavelengths)
        targets = targets + light_source_spectra_LED
    if HPS:
        light_source_spectra_HPS = load_and_resample_light_spectra('./data/light_spectra/high_pressure_sodium.csv', wavelengths)
        targets = targets + light_source_spectra_HPS
    if LPS:
        light_source_spectra_LPS = load_and_resample_light_spectra('./data/light_spectra/low_pressure_sodium.csv', wavelengths)
        targets = targets + light_source_spectra_LPS
    if MV:
        light_source_spectra_MV = load_and_resample_light_spectra('./data/light_spectra/mercury_vapor.csv', wavelengths)
        targets = targets + light_source_spectra_MV
    if FL:
        light_source_spectra_FL = load_and_resample_light_spectra('./data/light_spectra/fluorescent.csv', wavelengths)
        targets = targets + light_source_spectra_FL

    # Plot filter spectra
    fig = plt.figure()
    for target in targets:
        plt.plot(wavelengths, target['spectrum'], label=target['description'])# / max(target['spectrum']), label=target['description'])
    plt.legend()
    plt.show()
    plt.savefig('./figs/target_spectra.png', dpi=600)
    plt.close(fig)

    return targets


def load_and_resample_light_spectra(csv_path, resampling_wavelengths):
    data = load_csv(csv_path)
    data[1:, :] = (data[1:, :])  # the csv file is read as strings, so cast to float here
    lightsource_wavelengths = data[1:, 0].astype(float)
    spectra = data[1:, 1:].astype(float)
    descriptions = np.squeeze(data[:1, 1:])
    resampled_spectra = []
    resample = spectral.BandResampler(lightsource_wavelengths, resampling_wavelengths)

    for i in range(len(spectra[0, :])):
        spectrum = spectra[:, i]
        resampled = resample(spectra[:, i])
        result_dict = {}
        if not np.shape(descriptions):  # if shape is an empty tuple, the array only has one element: do not give index
            result_dict['description'] = descriptions
        else:
            result_dict['description'] = descriptions[i]
        result_dict['spectrum'] = resampled
        resampled_spectra.append(result_dict)

    return resampled_spectra


def load_EnMAP_L1B_data_with_EnPT(folder_path, type='vnir', indices_x=None, indices_y=None):
    """Loading an EnMAP image using the EnPT library. Optionally cropping the image by giving the x and y indices
    that should be kept. This can not deal with L1C data."""

    # Data is always located in a .zip file, within folder structure:
    # ./data/EnMAP/[folder for the datatake, given as parameter]/ENMAP.HSI.L1C/[observation name]/[filename].zip
    folder_path = Path(folder_path, 'ENMAP.HSI.L1B')
    observation_name = os.listdir(folder_path)[0]  # only the one subfolder in this directory
    folder_path = Path(folder_path, observation_name)
    name_list = os.listdir(folder_path)  # pick the only .zip file in this directory
    for filename in name_list:
        if '.ZIP' in filename:
            zip_name = filename
    path = Path(folder_path, zip_name)
    path = os.path.abspath(path)

    config_minimal = dict(
        path_l1b_enmap_image=path
    )

    CTR = EnPT_Controller(**config_minimal)
    CTR.read_L1B_data()

    # CTR.L1_obj.correct_dead_pixels()  # Running this says that over 90 percent of VNIR pixels are defective!
    vnir_cube = CTR.L1_obj.vnir.data.arr
    vnir_wavelengths = CTR.L1_obj.meta.vnir.wvl_center
    swir_cube = CTR.L1_obj.swir.data.arr
    swir_wavelengths = CTR.L1_obj.meta.swir.wvl_center

    if type == 'vnir':
        data = vnir_cube
        wavelengths = vnir_wavelengths
    elif type == 'swir':
        data = swir_cube
        wavelengths = swir_wavelengths
    if indices_y is not None:
        data = data[indices_y[0]:indices_y[1], :, :]
    if indices_x is not None:
        data = data[:, indices_x[0]:indices_x[1], :]

    return data, wavelengths


def extract_envi_paths_from_EnMAP_folder_structure(folder_path):
    """Extracting paths of ENVI header and binary image, and EnMAP xml metadata file from the EnMAP product folder
    structure. """
    # Data is always located in a .zip file, within folder structure:
    # ./data/EnMAP/[folder for the datatake, given as parameter]/ENMAP.HSI.L1C/[observation name]/[filename].zip
    folder_path = Path(folder_path, 'ENMAP.HSI.L1C')
    observation_name = os.listdir(folder_path)[0]  # only the one subfolder in this directory
    folder_path = Path(folder_path, observation_name)
    name_list = os.listdir(folder_path)  # list will include one folder and one .zip file, pick the folder
    for filename in name_list:
        if '.ZIP' not in filename:
            folder_path = Path(folder_path, filename)
    # Out of contents of this final folder, pick out the ENVI header, with extension .HDR
    name_list = os.listdir(folder_path)
    for filename in name_list:
        if '.HDR' in filename:
            hdr_path = Path(folder_path, filename)
        elif '.BIL' in filename:
            cube_path = Path(folder_path, filename)
        elif 'METADATA.XML' in filename:
            metadata_path = Path(folder_path, filename)

    return hdr_path, cube_path, metadata_path


def load_EnMAP_data_with_envi(folder_path, type=None, indices_x=None, indices_y=None):
    """Opening an EnMAP image using the ENVI header and binary file provided with the product. Also uses the xml
    metadata file to correct for gain and offset, transforming the DNs into radiance units. """

    file_list = os.listdir(folder_path)
    if len(file_list) == 3:  # if the folder only has three items, those are the hdr, the cube as a binary, and metadata
        for file_name in file_list:
            if ('.hdr' or '.HDR') in file_name:
                hdr_path = Path(folder_path, file_name)
                cube_path = Path(folder_path, file_name[:-4])
            elif '.xml' in file_name or '.XML' in file_name:
                metadata_path = Path(folder_path, file_name)
    else:  # If the number of items is not three, execute excavation of paths from the folder structure
        hdr_path, cube_path, metadata_path = extract_envi_paths_from_EnMAP_folder_structure(folder_path)

    envi_img, datacube, wavelengths = open_envi_image(hdr_path, cube_path, plot_one_channel=False)
    # open_envi_image_with_gdal(cube_path, crop_coordinates_x=(373000, 373000 + 30*500), crop_coordinates_y=(3960000, 3960000 - 30*200))

    # Find the relevant things from the metadata file: gain values, offset values, FWHMs and center wavelengths of bands
    # mf = glob.glob(l0_od + '/ENMAP01-_____L0*METADATA.XML')[0]
    root = lxml.objectify.parse(metadata_path).getroot()
    spc = root.specific
    # bse = root['base']
    acrossOffNadir = spc.acrossOffNadirAngle
    acrossTrackOffNadirAngles = {
        'upper_left': float(acrossOffNadir.upper_left),
        'upper_right': float(acrossOffNadir.upper_right),
        'lower_right': float(acrossOffNadir.lower_left),
        'lower_left': float(acrossOffNadir.lower_right)
    }

    bc = spc.bandCharacterisation
    vecCWL, vecFWHM, vecOffset, vecGain = [], [], [], []
    for bandID in bc.bandID:
        vecCWL.append(float(bandID.wavelengthCenterOfBand))
        vecFWHM.append(float(bandID.FWHMOfBand))
        vecOffset.append(float(bandID.OffsetOfBand))
        vecGain.append(float(bandID.GainOfBand))

    # Transfer the DNs of the datacube into radiance units
    vecGain = np.asarray(vecGain)
    vecOffset = np.asarray(vecOffset)
    datacube = (datacube * vecGain) + vecOffset

    # Replace the possibly erroneous ENVI wavelength vector with the metadata band center wavelengths
    wavelengths = np.asarray(vecCWL)
    FWHMs = np.asarray(vecFWHM)
    envi_img.metadata['fwhm'] = vecFWHM
    envi_img.metadata['wavelength'] = vecCWL
    envi_img.metadata['acrossTrackOffNadirAngles'] = acrossTrackOffNadirAngles

    # Retrieve the indices of VNIR and SWIR
    vnir_channel_list = spc.vnirProductQuality.expectedChannelsList
    vnir_channel_list = list(vnir_channel_list.text.split(','))
    vnir_channel_indices = np.asarray([int(i) - 1 for i in vnir_channel_list])
    all_channel_indices = np.asarray(range(len(wavelengths)))
    swir_channel_indices = [i for i in all_channel_indices if i not in vnir_channel_indices]

    if type == 'vnir':
        datacube = datacube[:, :, vnir_channel_indices]
        wavelengths = wavelengths[vnir_channel_indices]
        FWHMs = FWHMs[vnir_channel_indices]
    elif type == 'swir':
        datacube = datacube[:, :, swir_channel_indices]
        wavelengths = wavelengths[swir_channel_indices]
        FWHMs = FWHMs[swir_channel_indices]
    # if type is something else, just return the whole combined cube

    if indices_y is not None:
        datacube = datacube[indices_y[0]:indices_y[1], :, :]
    if indices_x is not None:
        datacube = datacube[:, indices_x[0]:indices_x[1], :]

    # Clip the data to get rid of negative radiances
    clip_value = 0.0
    datacube = np.clip(datacube, a_min=clip_value, a_max=1000) - clip_value

    return envi_img, datacube, wavelengths, FWHMs


def open_envi_image(hdr_path, cube_path, plot_one_channel=False):
    """Opening a spectral image saved in the ENVI format with a header file and a corresponding binary image. """
    img = spectral.envi.open(hdr_path, cube_path)

    datacube = np.asarray(img.asarray())
    try:
        wavelengths = np.asarray(img.metadata['wavelength']).astype(float)
    except:
        wavelengths = None

    if plot_one_channel:
        plt.figure()
        plt.imshow(datacube[:, :, int(len(wavelengths) / 4)])
        plt.show()

    return img, datacube, wavelengths


# TODO This is broken and awful, better to crop the images using QGIS. Consider removing the function, or fixing it.
def open_envi_image_with_gdal(cube_path, crop_coordinates_x=None, crop_coordinates_y=None):
    from osgeo import gdal
    data = gdal.Open(str(cube_path))
    geotransform = data.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]

    if crop_coordinates_x is not None and crop_coordinates_y is not None:
        upper_left_x = crop_coordinates_x[0]
        upper_left_y = crop_coordinates_y[0]
        lower_right_x = crop_coordinates_x[1]
        lower_right_y = crop_coordinates_y[1]
        # lower_right_x = originX + 100 * pixelWidth
        # lower_right_y = originY + 100 * pixelHeight
        window = (upper_left_x, upper_left_y, lower_right_x, lower_right_y)

        gdal.Translate(str(cube_path) + '_cropped.tif', str(cube_path), projWin=window)
        data = gdal.Open(str(cube_path) + '_cropped.tif')

    band = data.GetRasterBand(1)
    cols = data.RasterXSize
    rows = data.RasterYSize
    image_array = band.ReadAsArray(0, 0, cols, rows)

    plt.figure()
    plt.imshow(image_array)
    plt.show()

    print('test')
