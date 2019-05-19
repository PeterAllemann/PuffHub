"""
Handles the feature extraction.
"""

import numpy as np

IMAGE_PATH = "../Cropped-images/"
BLACK = 0
IMG_HEIGHT = 100


# get index of the lower contour
def lower_contour(x):
    black_pixels = np.where(x == BLACK)

    if len(black_pixels[0]) > 0:
        return black_pixels[0][-1]

    return len(x - 1)


# get index of the upper contour
def upper_contour(x):
    black_pixels = np.where(x == BLACK)

    if len(black_pixels[0]) > 0:
        return black_pixels[0][0]
    return 0


# calculate number of black pixels between lower and upper contour
def black_pixels_lc_uc(x):
    lc = lower_contour(x)
    uc = upper_contour(x)
    black_pixels = len(np.where(x[uc + 1: lc] == BLACK)[0])

    if black_pixels > 0:
        return black_pixels  # / ((lc) - (uc+1))

    return 0


# get number of black pixels
def nbr_black_pixels(x):
    return len(x[x == BLACK])  # /len(x)


#  calculte #b/w transitions
def black_white_transitions(x):
    counter = 0
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            counter = counter + 1
    return counter


def normalize_features(feature_vector, nbr_features, z=None):
    if z is None:

        contour_length = (feature_vector[0::nbr_features] - feature_vector[1::nbr_features])

        contour_length[contour_length < 1] = 1
        feature_vector[2::nbr_features] /= contour_length

        feature_vector[3::nbr_features] /= IMG_HEIGHT
        feature_vector[1::nbr_features] /= (IMG_HEIGHT - 1)
        feature_vector[0::nbr_features] /= (IMG_HEIGHT - 1)
        feature_vector[4::nbr_features] /= (IMG_HEIGHT - 1)

        return feature_vector

    else:
        for i, (mean, std) in enumerate(z):
            feature_vector[i::nbr_features] = (feature_vector[i::nbr_features] - mean) / std

        return feature_vector


# extract features for an image per column
def extract_features(img):
    img = img.resize((100, 100))
    img = np.array(img)
    f = []

    for c in img.T:
        f.append(lower_contour(c))
        f.append(upper_contour(c))
        f.append(black_pixels_lc_uc(c))
        f.append(nbr_black_pixels(c))
        f.append(black_white_transitions(c))

    return np.asarray(f, dtype=float)
