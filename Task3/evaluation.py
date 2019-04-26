"""
Handles the evaluation. It calculates the tp, fp, fn, tn and the accuracy.
It prints and stores everything into a text file.
"""

IMG_PATH = "Cropped-images/"
IMG_ENDING = ".png"
TRANSCRIPTION_PATH = './ground-truth/transcription.txt'
KEYWORDS = "task/keywords.txt"

tp_total = 0
fp_total = 0
fn_total = 0


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


# def get_accuracy(tp, total_words):
#     return tp/total_words


def evaluation(spotted_dict, test_samples, total_words, keyword):
    # compute FP, FN, TP for each image
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

    # update total FP, FN, TP
    global tp_total, fp_total, fn_total
    tp_total = tp_total + tp
    fp_total = fp_total + fp
    fn_total = fn_total + fn

    # print("\n", keyword)
    print("TP: ", tp)
    print("FP: ", fp)
    print("FN: ", fn)
    print("Recall: ", get_recall(tp, fn))
    print("Precision: ", get_precision(tp, fp))
    # print("Accuracy: ", get_accuracy(tp, total_words))
    print("---------------------------------------------------")


def print_final_statistics(top_n):
    print("\nTotal Recall & Precision:")
    print("N:", top_n)
    print("Recall:", get_recall(tp_total, fn_total))
    print("Precision:", get_precision(tp_total, fp_total))
