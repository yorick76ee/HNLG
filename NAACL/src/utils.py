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
import nltk
from sumeval.metrics.rouge import RougeCalculator

rouge = RougeCalculator(stopwords=True, lang="en")


# Checker
def check_dir(dir_path):
    ans = input(
            "Are you going to delete all old files in {} ? [y/n] (y) "
            .format(dir_path))
    if ans == 'n':
        exit()
    else:
        files = glob.glob(os.path.join(dir_path, "*"))
        for f in files:
            os.remove(f)


def check_usage():
    output = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv | tail -1',
            shell=True)
    free_mem = int(output.decode("utf-8").split(' ')[0])
    unit = output.decode("utf-8").split(' ')[1].strip()
    print_time_info("Free memory: {} {}".format(free_mem, unit))


# Time
def get_time():
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    return Y, M, D, h, m, s


def print_time_info(string):
    Y, M, D, h, m, s = get_time()
    _string = re.sub('[ \n]+', ' ', string)
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(
        Y, M, D, h, m, s, _string))


def print_curriculum_status(layer):
    print("###################################")
    print("#    CURRICULUM STATUS: LAYER{}    #".format(layer))
    print("###################################")


# Metric helper
def subseq(seq):
    # get subseq before the first _EOS token
    index_list = np.where(seq == _EOS)[0]
    if len(index_list) > 0:
        return seq[:index_list[0]]
    else:
        return seq


def BLEU(labels, hypothesis):
    # calculate the average BLEU score, trim paddings in labels
    bleu_score = [nltk.translate.bleu_score.sentence_bleu(
            [list(filter(lambda x: x != _PAD and x != _EOS, subseq(r)))],
            list(filter(lambda x: x != _PAD and x != _EOS, subseq(h)))
        )
        for r, h in np.stack(
            (labels, hypothesis), axis=1)
    ]

    return bleu_score


def ROUGE(labels, hypothesis):
    # [[rouge_1, rouge_2, rouge_l, rouge_be], ...]
    # use the subseq before the first _EOS token
    rouge_score = [
        [
            rouge.rouge_n(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=1
            ),
            rouge.rouge_n(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=2
            ),
            rouge.rouge_l(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))])
            ),
            rouge.rouge_be(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))])
            )
        ]
        for r, h in np.stack(
            (labels, hypothesis), axis=1)
    ]

    return rouge_score


# Argument helper
def add_path(args):
    args.embeddings_dir = os.path.join(args.data_dir, "GloVe")
    args.model_dir = os.path.join(args.data_dir, "model")
    args.log_dir = os.path.join(args.data_dir, "log")
    args.train_data_file = os.path.join(
            args.data_dir, "{}_train_data.pkl".format("E2ENLG"))
    args.valid_data_file = os.path.join(
            args.data_dir, "{}_valid_data.pkl".format("E2ENLG"))
    args.vocab_file = os.path.join(
            args.data_dir, "{}_vocab.pkl".format("E2ENLG"))
    args.data_dir = os.path.join(args.data_dir, "E2ENLG")
    return args


def print_config(args):
    print()
    print("{}:".format("PATH"))
    print("\t{}: {}".format("Data directory", args.data_dir))
    print("\t{}: {}".format("Embeddings directory", args.embeddings_dir))
    print("\t{}: {}".format("Model directory", args.model_dir))
    print("\t{}: {}".format("Log directory", args.log_dir))
    print("\t{}: {}".format("Processed train data file", args.train_data_file))
    print("\t{}: {}".format("Processed valid data file", args.valid_data_file))
    print("\t{}: {}".format("Processed vocab file", args.vocab_file))
    print("{}:".format("DATA"))
    print("\t{}: {}".format("Dataset", "E2ENLG"))
    print("\t{}: {}".format("Vocab size", args.vocab_size))
    print("\t{}: {}".format(
        "Use pretrained embeddings", bool(args.use_embedding)))
    print("{}:".format("MODEL"))
    print("\t{}: {}".format("Decoder layers", args.n_layers))
    print("\t{}: {}".format("RNN layers of encoder", args.n_en_layers))
    print("\t{}: {}".format("RNN layers of decoder", args.n_de_layers))
    print("\t{}: {}".format("Hidden size of encoder RNN", args.en_hidden_size))
    print("\t{}: {}".format("Hidden size of decoder RNN", args.de_hidden_size))
    print("\t{}: {}".format("Shared embedding dimension", args.embedding_dim))
    print("\t{}: {}".format("Attention method", args.attn_method))
    print("\t{}: {}".format("Bidrectional RNN", bool(args.bidirectional)))
    print("{}:".format("TRAINING"))
    print("\t{}: {}".format("Training epochs", args.epochs))
    print("\t{}: {}".format("Batch size", args.batch_size))
    print("\t{}: {}".format("Encoder optimizer", args.en_optimizer))
    print("\t{}: {}".format("Encoder learning_rate", args.en_learning_rate))
    print("\t{}: {}".format("Decoder optimizer", args.de_optimizer))
    print("\t{}: {}".format("Decoder learning rate", args.de_learning_rate))
    print("\t{}: {}".format(
        "Inner teacher forcing ratio", args.inner_teacher_forcing_ratio))
    print("\t{}: {}".format(
        "Inter teacher forcing ratio", args.inter_teacher_forcing_ratio))
    print("\t{}: {}".format(
        "Inner teacher forcing decay rate", args.inner_tf_decay_rate))
    print("\t{}: {}".format(
        "Inter teacher forcing decay rate", args.inter_tf_decay_rate))
    print("\t{}: {}".format("Curriculum learning", bool(args.is_curriculum)))
    print("\t{}: {}".format("Max gradient norm", args.max_norm))
    print("{}:".format("VERBOSE, VALIDATION AND SAVE"))
    print("\t{}: {}".format("Verbose epochs", args.verbose_epochs))
    print("\t{}: {}".format("Verbose batches", args.verbose_batches))
    print("\t{}: {}".format("Validation epochs", args.valid_epochs))
    print("\t{}: {}".format("Validation batches", args.valid_batches))
    print("\t{}: {}".format("Save epochs", args.save_epochs))
