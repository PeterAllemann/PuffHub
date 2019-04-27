import os
from PIL import Image
from Task3.features import extract_features
from Task3.sample_service import get_train_sample, get_test_sample
from Task3.evaluation import evaluation

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

IMG_PATH = "Cropped-images/"
KEYWORDS = "task/keywords.txt"
IMG_ENDING = ".png"


def calculate_test_features():
    dict = {}
    print("Start calculating features")
    for filename in os.listdir(IMG_PATH):
        if int(filename[0:3]) >= 300:
            img = Image.open(IMG_PATH + filename)
            dict[filename] = extract_features(img)

    print("End calculating features")
    return dict


def keyword_spotter(top_n):
    f = open(KEYWORDS, 'r')
    keywords = f.readlines()
    f.close()
    feature_dict = calculate_test_features()
    i = 1
    for k in keywords:
        print('keyword {} ({}/{})'.format(k, i, len(keywords)))
        train_sample = get_train_sample(k)
        test_sample = get_test_sample(k)

        img = Image.open(IMG_PATH + train_sample + IMG_ENDING)
        x = extract_features(img)

        dist_dict = {}
        total_words = 0

        for filename in os.listdir(IMG_PATH):
            if int(filename[0:3]) >= 300:
                total_words += 1
                y = feature_dict[filename]

                # start1 = time.time()

                # dist = dtw(x, y)

                # end1 = time.time()
                # print("time1:", end1-start1)
                #
                # start2 = time.time()

                dist, path = fastdtw(x, y, dist=euclidean)

                # end2 = time.time()
                # print("time2:", end2-start2)

                if len(dist_dict) < top_n:
                    dist_dict[filename[0:9]] = dist

                else:
                    if dist_dict[max(dist_dict, key=dist_dict.get)] > dist:
                        del dist_dict[max(dist_dict, key=dist_dict.get)]
                        dist_dict[filename[0:9]] = dist

        # print dist_dict
        evaluation(dist_dict, test_sample, k)
        i = i+1
