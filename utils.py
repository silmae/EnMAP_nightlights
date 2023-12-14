import numpy as np
import skimage.draw
import scipy
from matplotlib import pyplot as plt


def apply_elliptical_mask(data: np.ndarray, ellipse_parameters, masking_value=0) -> np.ndarray:
    """
    Create a circular mask and apply it to an image cube. Works for ndarrays and torch tensors.
    Mask creation from https://stackoverflow.com/a/44874588
    :param data:
        Data cube to be masked
    :param ellipse_parameters:
        List of four integers: center y-coordinate, center x-coordinate, height, width
    :param masking_value:
        Value to be used to replace the masked values. Default is 0, other useful values are np.nan (or float('nan')) and 1.
    :return:
        Masked datacube
    """

    # Draw ellipse
    rr, cc = skimage.draw.ellipse(ellipse_parameters[0], ellipse_parameters[1], ellipse_parameters[2], ellipse_parameters[3], data[:, :, 0].shape)
    # Make copy of data and overlay the mask
    masked_data = np.array(data)
    masked_data[rr, cc, :] = masking_value

    # # Plot to check mask shape
    # plt.figure()
    # plt.imshow(masked_data[:, :, 0])
    # plt.show()

    return masked_data


def find_nearest(array, value):
    "Element in nd array `array` closest to the scalar value `value`"
    idx = np.abs(array - value).argmin()
    return idx


def interpolate_offNadir_angle(data, envi_img, coordinates):
    for i in range(data.shape[0]):
        if data[i, 1, 0] - data[0, 1, 0] != 0:
            y_index = i + 1
            break

    for i in range(data.shape[0]):
        if data[1, i, 0] - data[1, 0, 0] != 0:
            x_index = i + 1
            break

    scene_corner_coordinates = {
        'upper_left': (y_index, 0),
        'upper_right': (0, x_index),
        'lower_left': (data.shape[0], data.shape[1] - x_index),
        'lower_right': (data.shape[0] - y_index, data.shape[1])
    }

    angles = envi_img.metadata['acrossTrackOffNadirAngles']
    points = np.zeros((4, 2))
    points[0, :] = scene_corner_coordinates['upper_left']
    points[1, :] = scene_corner_coordinates['upper_right']
    points[2, :] = scene_corner_coordinates['lower_left']
    points[3, :] = scene_corner_coordinates['lower_right']

    # # Sanity check plot
    # plt.figure()
    # plt.imshow(np.sum(data, axis=2) ** 0.2)
    # plt.scatter(points[:, 1], points[:, 0])
    # plt.scatter(coordinates[1], coordinates[0])
    # plt.show()

    values = np.zeros((4, 1))
    values[0] = angles['upper_left']
    values[1] = angles['upper_right']
    values[2] = angles['lower_left']
    values[3] = angles['lower_right']

    interp_angle = scipy.interpolate.griddata(points=points, values=values, xi=coordinates)

    return np.squeeze(interp_angle)