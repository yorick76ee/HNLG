from torch.utils.data import Dataset
import os
import pickle
import json
import numpy as np
from utils import print_time_info
from tokenizer import Tokenizer
from text_token import _UNK, _PAD, _BOS, _EOS
from data.E2ENLG import E2ENLG


class DataEngine(Dataset):
    def __init__(
            self,
            data_dir,
            save_path,
            vocab_path,
            vocab_size,
            n_layers,
            regen,
            train
    ):
        self.data_dir = data_dir
        self.save_path = save_path
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.min_length = 5
        self.regen = regen
        self.split_vocab = True
        self.tokenizer = Tokenizer(vocab_path, self.split_vocab, regen, train)
        self.train = train
        self.prepare_data()

    def prepare_data(self):
        if not os.path.exists(self.save_path) or self.regen:
            if self.regen:
                print_time_info("Regenerate the data...")
            else:
                print_time_info("There isn't any usable save...")
            if not os.path.isdir(self.data_dir):
                print_time_info("Error: The dataset doesn't exist")
                exit()
            print_time_info("Start reading dataset E2ENLG from {}".format(
                self.data_dir))

            self.input_data, self.output_labels = E2ENLG(
                    self.data_dir, self.min_length, self.train)
        else:
            self.input_data, self.output_labels = \
                    pickle.load(open(self.save_path, 'rb'))
            print_time_info("Load the data from {}".format(self.save_path))

        if not os.path.exists(self.vocab_path) or (self.regen and self.train):
            self.build_vocab()
        if not os.path.exists(self.save_path) or self.regen:
            self.tokenize_sents()
            pickle.dump(
                    [self.input_data, self.output_labels],
                    open(self.save_path, 'wb'))
            print_time_info(
                    "Create the save file {}".format(self.save_path))

        # shrink the vocab to vocab size
        self.tokenizer.shrink_vocab(self.vocab_size)
        self.add_unk()

        # pick the labels for different n_layers
        if self.n_layers == 1:
            self.output_labels = [self.output_labels[3]]
        elif self.n_layers == 2:
            self.output_labels = [self.output_labels[1], self.output_labels[3]]

    def build_vocab(self):
        if not self.split_vocab:
            corpus = []
            for sent in self.input_data:
                corpus.extend(sent)
            for sent in self.output_labels[-1]:
                corpus.extend(sent)
            self.tokenizer.build_vocab(corpus)
        else:
            corpus = []
            for sent in self.output_labels[-1]:
                corpus.extend(sent)
            tokens = []
            for attrs in self.input_data:
                tokens.extend(attrs)
            self.tokenizer.build_vocab(corpus, tokens)

    def tokenize_sents(self):
        for idx, sent in enumerate(self.input_data):
            self.input_data[idx] = self.tokenizer.tokenize(sent, True)
        for idx, output_labels in enumerate(self.output_labels):
            for sidx, sent in enumerate(output_labels):
                self.output_labels[idx][sidx] = \
                        self.tokenizer.tokenize(sent, False)
                self.output_labels[idx][sidx].append(_EOS)

    def add_unk(self):
        if not self.split_vocab:
            for idx, sent in enumerate(self.input_data):
                for w_idx, word in enumerate(sent):
                    # 4 for special token
                    if word >= self.vocab_size + 4:
                        self.input_data[idx][w_idx] = _UNK

        for l_idx, labels in enumerate(self.output_labels):
            for idx, sent in enumerate(labels):
                for w_idx, word in enumerate(sent):
                    # 4 for special token
                    if word >= self.vocab_size + 4:
                        self.output_labels[l_idx][idx][w_idx] = _UNK

    def expand(self):
        inter_labels = [[], [], [], []]
        for l_idx, labels in enumerate(self.output_labels[:-1]):
            for s_idx, sent in enumerate(labels):
                upper_sent = self.output_labels[l_idx+1][s_idx]
                inter_label = []
                lidx, hidx = 0, 0
                while lidx != len(sent):
                    inter_label.append(sent[lidx])
                    if sent[lidx] == upper_sent[hidx]:
                        lidx += 1
                    hidx += 1
                inter_labels[l_idx+1].append(inter_label)

        for _ in range(len(self.input_data)):
            inter_labels[0].append([_UNK])

        return inter_labels

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)

    def untokenize(self, sent, is_token=False):
        return self.tokenizer.untokenize(sent, is_token)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (
            self.input_data[idx],
            [labels[idx] for labels in self.output_labels])
