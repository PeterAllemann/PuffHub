"""
Handles the evaluation. It calculates the tp, fp, fn, tn and the accuracy.
It prints and stores everything into a text file.
"""
from Task3.keyword_spotter import get_similar_images
from Task3.service import load_train_word_dict, load_valid_word_dict, \
    load_keywords, get_train_sample, load_valid_words, load_test_words, load_keywords_for_test, \
    get_train_sample_for_test

IMG_PATH = "Cropped-images/"
IMG_ENDING = ".png"
TRANSCRIPTION_PATH = './ground-truth/transcription.txt'
RESULTS_PATH = 'results.txt'

VALID_WORDS = load_valid_words()
TEST_WORDS = load_test_words()

TRAIN_WORD_DICT = load_train_word_dict()
VALID_WORD_DICT = load_valid_word_dict()

KEYWORDS = load_keywords()
KEYWORDS_TEST = load_keywords_for_test()


def do_keyword_spotting():
    """Starts the keyword spotting."""
    for keyword in KEYWORDS_TEST:
        train_word = get_train_sample_for_test(keyword)
        result = get_similar_images(train_word, TEST_WORDS)
        print('result: {}'.format(result))
        # tpt, fpt, tnt, fnt = get_confusion_matrix(keyword, result)
        # print_results(tpt, fpt, tnt, fnt)
        save_results(keyword, result)


def get_confusion_matrix(keyword, retrieved_words):
    """Calculates the confusion matrix."""
    true_positives = get_tp(keyword, retrieved_words)
    false_positives = len(retrieved_words) - true_positives
    false_negatives = get_fn(keyword) - true_positives
    true_negatives = len(VALID_WORDS) - true_positives - false_negatives - false_positives
    return true_positives, false_positives, true_negatives, false_negatives


def get_tp(keyword, retrieved_words):
    """Calculate true positives."""
    count = 0
    for word in retrieved_words:
        if VALID_WORD_DICT[word[1]] == keyword:
            count += 1
    return count


def get_fn(keyword):
    """Calculate false negatives."""
    count = 0
    for word in VALID_WORDS:
        if VALID_WORD_DICT[word] == keyword:
            count += 1
    return count


def save_results(keyword, result):
    """"Saves the results in a result file."""
    file = open(RESULTS_PATH, "a")
    out = keyword
    for tuple_ in result:
        out = out + ', ' + tuple_[1] + ', ' + str(tuple_[0])
    print(out, file=file)
    file.close()


def print_results(true_positives, false_positives, true_negatives, false_negatives):
    """Prints the results."""
    precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives == 0:
        recall = None
    else:
        recall = true_positives / (true_positives + false_negatives)
    accuracy = true_positives / (true_positives + false_positives + false_negatives + true_negatives)

    print("Precision", precision, "recall", recall, "Accuracy", accuracy)
