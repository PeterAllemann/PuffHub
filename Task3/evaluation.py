"""
Handles the evaluation. It calculates the tp, fp, fn, tn and the accuracy.
It prints and stores everything into a text file.
"""

import os
from PIL import Image
from Task3.feature_extraction.features import extract_features
from Task3.DTW.DTW import dtw

IMG_PATH = "Cropped-images/"
IMG_ENDING = ".png"
TRANSCRIPTION_PATH = './ground-truth/transcription.txt'
KEYWORDS = "task/keywords.txt"

TOP_N = 50

tp_total = 0
fp_total = 0
fn_total = 0


def calculate_test_features():
    dict = {}
    print("Start calculating features")
    for filename in os.listdir(IMG_PATH):
        if int(filename[0:3]) >= 300:
            img = Image.open(IMG_PATH + filename)
            dict[filename] = extract_features(img)

    print("End calculating features")
    return dict


def get_recall(tp, fn):
    if (tp + fn) == 0:
        return None
    else:
        return float(tp) / (tp + fn)


def get_precision(tp, fp):
    if (tp + fp) == 0:
        return None
    else:
        return float(tp) / (tp + fp)


def get_accuracy(tp, total_words):
    return tp/total_words


def evaluation(spotted_dict, test_samples, total_words, k):
    fp = 0
    fn = 0
    tp = 0

    total_samples = len(test_samples)
    for key in spotted_dict:
        if key in test_samples:
            tp = tp + 1
            total_samples = total_samples - 1
        else:
            fp = fp + 1

    fn = total_samples

    global tp_total, fp_total, fn_total
    tp_total = tp_total + tp
    fp_total = fp_total + fp
    fn_total = fn_total + fn

    print("\n", k)
    print("TP: ", tp)
    print("FP: ", fp)
    print("FN: ", fn)
    print("Recall: ", get_recall(tp, fn))
    print("Precision: ", get_precision(tp, fp))
    print("Accuracy: ", get_accuracy(tp, total_words))
    print()
    "****************************************************************************************************"


def keyword_spotter(topN):
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

                dist = dtw(x, y)

                if len(dist_dict) < topN:
                    dist_dict[filename[0:9]] = dist

                else:
                    if dist_dict[max(dist_dict, key=dist_dict.get)] > dist:
                        del dist_dict[max(dist_dict, key=dist_dict.get)]
                        dist_dict[filename[0:9]] = dist

        # print dist_dict
        evaluation(dist_dict, test_sample, total_words, k)
        i = i+1


def get_train_sample(keyword):
    f = open(TRANSCRIPTION_PATH, 'r')
    words = f.readlines()
    f.close()
    for w in words:
        if w[10:].lower() == keyword.lower() and int(w[0:3]) < 300:
            # print w
            return w[0:9]


def get_test_sample(keyword):
    d = {}
    f = open(TRANSCRIPTION_PATH, 'r')
    words = f.readlines()
    f.close()
    for w in words:
        if w[10:].lower() == keyword.lower() and int(w[0:3]) >= 300:
            d[w[0:9]] = keyword
    return d


keyword_spotter(TOP_N)
print("\nTotal Recall & Precision:")
print("N:", TOP_N)
print("Recall:", get_recall(tp_total, fn_total))
print("Precision:", get_precision(tp_total, fp_total))
