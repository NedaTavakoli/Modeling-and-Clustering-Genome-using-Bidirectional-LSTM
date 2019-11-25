# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import collections
import os

import numpy as np


class Words:
    def __init__(self, preprocess_dna):
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.words = None
        self.preprocess_dna = preprocess_dna

    def _build_vocab(self, filename):
        with open(filename, "r") as f:  # Changed to use open instead of tf.gfile.GFile so tensorFlow not needed.
            read_text = f.read()
        data = self.text_to_words(read_text)

        counter = collections.Counter(data)  # Word count of each word.
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  # Descending sorted list of tuples: (word, word ct)

        self.words, _ = list(zip(*count_pairs))  # List of words in descending order by word count. (no word count)
        self.word_to_id = dict(zip(self.words, range(len(self.words))))  # Dictionary word:
        self.id_to_word = dict((v, k) for k, v in self.word_to_id.items())

    def _file_to_word_ids(self, filename):
        """
        Takes a text file and converts all the text to a list of word_ids
        :param filename: The location of the text file to convert
        :return: list: A list of word ids
        """
        with open(filename, "r") as f:
            read_text = f.read()
        read_text
        words = self.text_to_words(read_text)
        word_ids = self.words_to_word_ids(words)
        return word_ids

    def words_to_word_ids(self, words):
        """
        Converts an iterable of words to a list of word ids
        :param words: An iterable of string words
        :return: word_ids: list: A list of word ids
        """
        word_ids = [self.word_to_id[word] for word in words if word in self.word_to_id]
        return word_ids

    def word_ids_to_words(self, word_ids):
        """
        Converts an iterable of word ids to a list of words
        :param word_ids: an iterable of word ids
        :return: words: list: A list of words
        """
        words = [self.id_to_word[word] for word in word_ids if word in self.id_to_word]
        return words

    def ptb_raw_data(self, data_path=None, prefix=""):
        """Load PTB raw data from data directory "data_path".
        Reads PTB text files, converts strings to integer ids,
        and performs mini-batching of the inputs.
        The PTB dataset comes from Tomas Mikolov's webpage:
        http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
        Args:
          data_path: string path to the directory where simple-examples.tgz has
            been extracted.
        Returns:
          tuple (train_data, valid_data, test_data)
          where each of the data objects can be passed to PTBIterator.
        """

        train_path = data_path + prefix + ".train.txt"
        valid_path = data_path + prefix + ".valid.txt"
        test_path = data_path + prefix + ".test.txt"

        self._build_vocab(train_path)  # Creates word to id and id to word dictionaries
        train_data = self._file_to_word_ids(train_path)  # Turns words to word ids. Training
        # I don't see the point of validation set
        # valid_data = self._file_to_word_ids(valid_path)  # Turns words to word ids. Validation
        valid_data = None
        test_data = self._file_to_word_ids(test_path)  # Turns words to word ids. Validation

        return train_data, valid_data, test_data

    def text_to_words(self, read_text):
        """
        Takes a string of words and breaks it into a list of words (each word being a string)
        :param read_text: a string of words
        :return: read_text: list: A list of words
        """
        if self.preprocess_dna is not None:  # Break up characters of DNA sequence using set word length
            read_text = self.process_dna(read_text, remove_short=True)
        else:  # Break up the words normally
            read_text = read_text.replace("\n", " ")
            read_text = read_text.split()
        return read_text

    def process_dna(self, input_string, remove_short):
        """
        Removes whitespace then breaks a DNA sequence up into strings of a set length (similar to words)
        :param input_string: string: a string of DNA
        :param remove_short: boolean: Whether to remove the last sequence if it is below the desired length.
        :return: working: list: list of DNA strings each self.preprocess_dna long.
        """
        # Shouldn't be any spaces or newlines but remove anyway
        working = input_string.replace(" ", "").replace("\n", "")
        working = [working[i:i + self.preprocess_dna] for i in range(0, len(working), self.preprocess_dna)]
        if remove_short and len(working[-1]) != self.preprocess_dna:
            working.pop()  # Remove last word if it is short
        # working = " ".join(working)  Not needed

        return working


def ptb_iterator(raw_data, batch_size, num_steps, direction=None):
    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, threpackage_hiddene batch size.
      num_steps: int, the number of unrolls.
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)  # Convert list to np array (word ids of words).

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]  # The end of the raw data doesn't go into a batch it doesn't divide evenly

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]  # Think this is the input to the LSTM
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]  # Think this is the expected out
        if direction == 'backward':
            temp = np.flip(x, axis=1)
            x = np.flip(y, axis=1)
            y = temp
        yield (x, y)
