"""
Gets the train sample and the test samples for a specific keyword.
"""

TRANSCRIPTION_PATH = './ground-truth/transcription.txt'


def get_train_sample(keyword):
    f = open(TRANSCRIPTION_PATH, 'r')
    words = f.readlines()
    f.close()
    for w in words:
        if w[10:].lower() == keyword.lower() and int(w[0:3]) < 300:
            return w[0:9]


def get_test_samples(keyword):
    d = {}
    f = open(TRANSCRIPTION_PATH, 'r')
    words = f.readlines()
    f.close()
    for w in words:
        if w[10:].lower() == keyword.lower() and int(w[0:3]) >= 300:
            d[w[0:9]] = keyword
    return d
