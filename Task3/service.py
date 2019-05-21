"""Handles the data."""
import os

TRANSCRIPTION_PATH = './ground-truth/transcription.txt'
KEYWORDS = "task/keywords_train.txt"
KEYWORDS_TEST = "task/keywords.txt"
IMG_PATH = "Cropped-images/"


def get_train_sample(keyword):
    """Gets a train image that matches the keyword."""
    file = open(TRANSCRIPTION_PATH, 'r')
    words = file.readlines()
    file.close()
    for word in words:
        word = word.rstrip('\n')
        if word[10:].lower() == keyword.lower() and int(word[0:3]) < 300:
            return word[0:9]


def get_train_sample_for_test(keyword):
    """Gets a train image that matches the keyword."""
    file = open(KEYWORDS_TEST, 'r')
    lines = file.readlines()
    file.close()
    train_sample = ''
    for line in lines:
        transcript, word_id = str.split(line, ",")
        if transcript == keyword:
            train_sample = word_id.rstrip('\n')
    return train_sample


def load_valid_words():
    """Loads the validation words."""
    valid_word_list = []
    file = open(TRANSCRIPTION_PATH, 'r')
    words = file.readlines()
    file.close()
    for word in words:
        if int(word[0:3]) >= 300:
            valid_word_list.append(word[0:9])
    return valid_word_list


def load_test_words():
    """Loads the validation words."""
    return [f.rstrip('.png\n') for f in os.listdir(IMG_PATH) if
            ((os.path.splitext(f)[-1] == '.png') & (int(f[0:3]) > 304))]


def load_train_word_dict():
    """Creates a dictionary for the train words."""
    train_dict = {}
    with open(TRANSCRIPTION_PATH) as file:
        for line in file:
            if int(line[0:3]) < 300:
                word_id, transcript = str.split(line, " ")
                train_dict[word_id] = transcript.rstrip('\n')
    return train_dict


def load_valid_word_dict():
    """Creates a dictionary for the validation words."""
    valid_dict = {}
    with open(TRANSCRIPTION_PATH) as file:
        for line in file:
            if int(line[0:3]) >= 300:
                word_id, transcript = str.split(line, " ")
                valid_dict[word_id] = transcript.rstrip('\n')
    return valid_dict


def load_keywords():
    """Loads the keywords."""
    keywords = []
    file = open(KEYWORDS, 'r')
    words = file.readlines()
    file.close()
    for word in words:
        keywords.append(word.rstrip('\n'))
    return keywords


def load_keywords_for_test():
    """Loads the keywords."""
    keywords = []
    file = open(KEYWORDS_TEST, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        transcript, word_id = str.split(line, ",")
        keywords.append(transcript.rstrip('\n'))
    return keywords
