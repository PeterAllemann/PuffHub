"""
Handles the keyword spotting.
"""

import os
from PIL import Image
from Task3.features import extract_features, normalize_features
from Task3.sample_service import get_train_sample, get_test_samples
from Task3.evaluation import evaluation

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

IMG_PATH = "Cropped-images/"
KEYWORDS = "task/keywords.txt"
IMG_ENDING = ".png"


def calculate_test_features():
    feature_dict = {}
    print("Start calculating features")
    for filename in os.listdir(IMG_PATH):
        if int(filename[0:3]) >= 300:
            img = Image.open(IMG_PATH + filename)
            features = extract_features(img)
            feature_dict[filename] = normalize_features(features, 5)

    print("End calculating features")
    return feature_dict


def keyword_spotter(top_n):
    f = open(KEYWORDS, 'r')
    keywords = f.readlines()
    f.close()
    feature_dict = calculate_test_features()
    i = 1
    for k in keywords:
        print('keyword {} ({}/{})'.format(k, i, len(keywords)))
        train_sample = get_train_sample(k)
        test_samples = get_test_samples(k)

        img = Image.open(IMG_PATH + train_sample + IMG_ENDING)
        features = extract_features(img)
        x = normalize_features(features, 5)

        dist_dict = {}
        total_words = 0

        for filename in os.listdir(IMG_PATH):
            if int(filename[0:3]) >= 300:
                total_words += 1
                y = feature_dict[filename]

                dist, path = fastdtw(x, y, dist=euclidean)

                if len(dist_dict) < top_n:
                    dist_dict[filename[0:9]] = dist

                else:
                    if dist_dict[max(dist_dict, key=dist_dict.get)] > dist:
                        del dist_dict[max(dist_dict, key=dist_dict.get)]
                        dist_dict[filename[0:9]] = dist

        evaluation(dist_dict, test_samples, k)
        i = i+1
