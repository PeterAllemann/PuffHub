"""
Handles the keyword spotting.
"""

from PIL import Image
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from Task3.features import extract_features, normalize_features

IMG_PATH = "Cropped-images/"
KEYWORDS = "task/keywords.txt"
IMG_ENDING = ".png"

PERCENTAGE = 0.9
LIMIT = 0.2


def calculate_distances(test_word, valid_words):
    """Calculates the distance"""
    distances = list()
    test_img = Image.open(IMG_PATH + test_word + IMG_ENDING)
    test_img_fv = normalize_features(extract_features(test_img), 5)
    for word in valid_words:
        train_image = Image.open(IMG_PATH + word + IMG_ENDING)
        train_image_fv = normalize_features(extract_features(train_image), 5)
        dist, _ = fastdtw(test_img_fv, train_image_fv, dist=euclidean)
        distances.append((dist, word))
    return distances


def get_threshold(distances):
    """Calculates the threshold."""
    distances = sorted(distances, key=lambda x: x[0])
    largest = 0
    counter = 0
    for i in range(1, len(distances)):
        if LIMIT > abs(distances[i][0] - distances[i - 1][0]):
            counter += 1
            if counter > 4:
                largest = distances[i][0]
                break
        else:
            counter = 0
    if largest == 0:
        largest = 40
    return (abs(largest - distances[0][0]) * PERCENTAGE) + distances[0][0]


def get_most_similar(distances):
    """Gets the most similar words."""
    threshold = get_threshold(distances)
    most_similar = list()
    distances = sorted(distances, key=lambda x: x[0])

    for i in distances:
        if i[0] < threshold:
            most_similar.append(i)
        else:
            break

    return most_similar


def get_similar_images(test_word, train_words):
    """Gets similar images."""
    distances = calculate_distances(test_word, train_words)
    return get_most_similar(distances)
