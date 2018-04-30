import time
import datetime
import math
import os
import sys
from text_token import _UNK, _PAD, _BOS, _EOS
import numpy as np
import glob
import subprocess
import torch
import re
from torch import optim


# Data loader
def collate_fn(batch):
    n_layers = len(batch[0][1])
    is_inter = (len(batch[0]) == 3)
    if is_inter:
        encoder_input, decoder_labels, inter_labels = \
            [], [[] for _ in range(n_layers)], [[] for _ in range(n_layers)]
    else:
        encoder_input, decoder_labels = [], [[] for _ in range(n_layers)]
    for data in batch:
        encoder_input.append(data[0])
        for idx in range(n_layers):
            decoder_labels[idx].append(data[1][idx])
            if is_inter:
                inter_labels[idx].append(data[2][idx])

    en_max_length = max([len(sent) for sent in encoder_input])
    de_max_lengths = [
            max([len(sent) for sent in labels]) for labels in decoder_labels]
    de_lengths = [
            sum(len(sent) for sent in labels) for labels in decoder_labels]

    encoder_input = pad_sequences(encoder_input, en_max_length, 'pre')
    for idx in range(n_layers):
        decoder_labels[idx] = \
                pad_sequences(decoder_labels[idx], de_max_lengths[idx], 'post')
    if is_inter:
        for idx in range(n_layers):
            inter_labels[idx] = \
                pad_sequences(inter_labels[idx], de_max_lengths[idx], 'post')
        return encoder_input, decoder_labels, inter_labels, de_lengths
    else:
        return encoder_input, decoder_labels, de_lengths


def pad_sequences(data, max_length, pad_type):
    if _PAD != -1:
        padded_data = np.full((len(data), max_length), _PAD)
    else:
        padded_data = np.full((len(data), max_length), _UNK)
    if pad_type == "post":
        for idx, d in enumerate(data):
            padded_data[idx][:min(max_length, len(d))] = \
                    d[:min(max_length, len(d))]
    elif pad_type == "pre":
        for idx, d in enumerate(data):
            padded_data[idx][max(0, max_length-len(d)):] = \
                    d[:min(max_length, len(d))]
    return padded_data


# Model helper
def get_embeddings(vocab, embeddings_dir, embedding_dim):
    embedding_file = os.path.join(
            embeddings_dir, "glove.6B.{}d.txt".format(embedding_dim))
    embeddings = torch.nn.Parameter(
            torch.Tensor(torch.randn(len(vocab), embedding_dim)))
    with open(embedding_file, 'r') as file:
        for line in file:
            data = line.strip().split(' ')
            word, emb = \
                data[0], torch.Tensor(np.array(list(map(float, data[1:]))))
            if word not in vocab:
                continue
            embeddings.data[vocab[word]] = torch.Tensor(emb)

    return embeddings


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer == "Adam":
        return optim.Adam(
                parameters, lr=learning_rate)
    elif optimizer == "RMSprop":
        return optim.RMSprop(
                parameters, lr=learning_rate)
    elif optimizer == "SGD":
        return optim.SGD(
                parameters, lr=learning_rate)
