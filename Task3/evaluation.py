"""
Handles the evaluation. It calculates the tp, fp, fn, tn and the accuracy.
It prints and stores everything into a text file.
"""

IMG_PATH = "Cropped-images/"
IMG_ENDING = ".png"
TRANSCRIPTION_PATH = './ground-truth/transcription.txt'
KEYWORDS = "task/keywords.txt"
RESULTS_PATH = 'results.txt'

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


def evaluation(spotted_dict, test_samples, keyword):
    # compute FP, FN, TP for each image
    fp = 0
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

    recall = get_recall(tp, fn)
    precision = get_precision(tp, fp)

    save_results(keyword, tp, fp, fn, recall, precision)
    print("TP: ", tp)
    print("FP: ", fp)
    print("FN: ", fn)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("---------------------------------------------------")


def print_final_statistics(top_n):
    recall = get_recall(tp_total, fn_total)
    precision = get_precision(tp_total, fp_total)

    save_final_statistics(top_n, recall, precision)

    print("\nTotal Recall & Precision:")
    print("N:", top_n)
    print("Recall:", recall)
    print("Precision:", precision)


def save_results(keyword, tp, fp, fn, recall, precision):
    out = "Keyword:" + format(keyword) + "\nTP: " + format(tp) + "\nFP: " + format(fp) + "\nFN: " + format(fn) + \
          "\nRecall: " + format(recall) + "\nPrecision: " + format(precision) + "\n----------------------------------\n"
    write_to_file(out)


def save_final_statistics(top_n, recall, precision):
    out = "\nTotal Recall & Precision\nN: " + format(top_n) + "\nRecall: " + format(recall) + "\nPrecision: " + \
          format(precision)
    write_to_file(out)


def write_to_file(out):
    with open(RESULTS_PATH, "a") as file:
        file.write(out)
