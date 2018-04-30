from nltk.corpus import wordnet, treebank
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
import os
from os.path import join as opj
import ast
import random
import pickle
import json
import numpy as np
import string
import functools
from copy import deepcopy
from tqdm import tqdm
from utils import print_time_info
from tokenizer import Tokenizer
from text_token import _UNK, _PAD, _BOS, _EOS


def E2ENLG(data_dir, min_length, train=True):
    # Step 1: Get the raw dialogues from data files
    raw_dialogues = parse_data(data_dir, train)
    # Step 2: Parse the dialogues
    dialogues = parse_dialogues(raw_dialogues)
    # Step 3: Build the input data and output labels
    input_data, output_labels = build_dataset(dialogues, min_length)
    return input_data, output_labels


def parse_data(data_dir, train):
    if train:
        lines_file = os.path.join(data_dir, "trainset.csv")
    else:
        lines_file = os.path.join(data_dir, "devset.csv")
    data = list()
    with open(lines_file, 'r') as file:
        for line in file:
            if len(line.split('"')) >= 3:
                attributes = line.split('"')[1].strip('"')
                s = line.replace("\"{}\",".format(attributes), "") \
                    .replace("\n", "")
                attributes = attributes.split(',')
                attributes = [
                        [
                            i.strip().split('[')[0],
                            i.strip().split('[')[1].strip(']')
                        ] for i in attributes]
                # trim all punctuation marks in one line
                seq = functools.reduce(
                        lambda s, c: s.replace(c, ''), string.punctuation, s)
                data.append([attributes, seq])

    for idx, d in enumerate(data):
        for a_idx, attr_pair in enumerate(d[0]):
            data[idx][0][a_idx][1] = attr_pair[1].replace("£", "")
        data[idx][1] = d[1].replace("£", "")
        data[idx][1] = data[idx][1].replace("2030", "20 30")
        data[idx][1] = data[idx][1].replace("2025", "20 25")
        data[idx][1] = data[idx][1].replace("2530", "25 30")

    for idx, d in enumerate(data):
        for a_idx, attr_pair in enumerate(d[0]):
            data[idx][0][a_idx][1] = attr_pair[1].lower()
        data[idx][1] = data[idx][1].lower()
        for attr_pair in d[0]:
            if attr_pair[0] in ['name', 'near']:
                if attr_pair[1] in data[idx][1]:
                    data[idx][1] = data[idx][1].replace(
                            attr_pair[1], attr_pair[0].upper()+"TOKEN")
                    continue
                # Some dirty rules below lol
                flag = False
                raw_attr = attr_pair[1]
                attr = raw_attr.split(' ')
                for a_idx in range(len(attr)):
                    attr[a_idx] = attr[a_idx][:-1]
                    if ' '.join(attr) in data[idx][1]:
                        flag = True
                        data[idx][1] = data[idx][1].replace(
                                ' '.join(attr),
                                attr_pair[0].upper()+"TOKEN")
                        break
                    attr = raw_attr.split(' ')
                if flag:
                    continue
                for a_idx in range(len(attr)):
                    attr[a_idx] = attr[a_idx] + ' '
                    if ' '.join(attr) in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                ' '.join(attr),
                                attr_pair[0].upper()+"TOKEN")
                        flag = True
                        break
                    attr = raw_attr.split(' ')
                if flag:
                    continue
                if attr_pair[1] == "fitzbillies":
                    if "fitzbilies" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "fitzbilies",
                                attr_pair[0].upper()+"TOKEN")
                if attr_pair[1] == "crowne plaza hotel":
                    if "crowne plaza taste of cambridge" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "crowne plaza taste of cambridge",
                                attr_pair[0].upper())
                    elif "crowne plaza" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "crowne plaza",
                                attr_pair[0].upper()+"TOKEN")
                if attr_pair[1] == "raja indian cuisine":
                    if "raja cuisine" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "raja cuisine",
                                attr_pair[0].upper()+"TOKEN")
                    elif "raja" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "raja", attr_pair[0].upper()+"TOKEN")
                if attr_pair[1] == "the portland arms":
                    if "portland arms" in data[idx][1]:
                        data[idx][1] = data[idx][1].replace(
                                "portland arms",
                                attr_pair[0].upper()+"TOKEN")

    for idx, d in enumerate(data):
        for a_idx, attr_pair in enumerate(d[0]):
            if attr_pair[0] in ['name', 'near']:
                data[idx][0][a_idx][1] = attr_pair[0].upper()

    for idx, d in enumerate(data):
        split_sent = d[1].split()
        for w_idx, w in enumerate(split_sent):
            if "NAMETOKEN" in w and "NEARTOKEN" in w:
                new_sent = []
                for nw in d[1][:w_idx]:
                    new_sent.append(nw)
                if w.find("NAMETOKEN") < w.find("NEARTOKEN"):
                    new_sent.extend(["NAMETOKEN", "NEARTOKEN"])
                else:
                    new_sent.extend(["NEARTOKEN", "NAMETOKEN"])
                for nw in d[1][w_idx+1:]:
                    new_sent.append(nw)
                data[idx][1] = ' '.join(new_sent)
                break

    for idx, d in enumerate(data):
        split_sent = d[1].split()
        new_sent = []
        for w_idx, w in enumerate(split_sent):
            if "NAMETOKEN" in w:
                new_sent.append("NAMETOKEN")
            elif "NEARTOKEN" in w:
                new_sent.append("NEARTOKEN")
            else:
                new_sent.append(w)
        data[idx][1] = ' '.join(new_sent)

    return data


def parse_dialogues(raw_dialogues):
    dialogues = []
    spacy_parser = spacy.load('en')
    for idx, dialog in enumerate(raw_dialogues):
        if idx % 1000 == 0:
            print_time_info(
                    "Processed {}/{} dialogues".format(
                        idx, len(raw_dialogues)))
        parsed_dialog = []
        # encoder input
        parsed_dialog.append(dialog[0])
        # output label
        line = dialog[1]
        parsed_line = spacy_parser(line)
        spacy_line = [
            d for d in [
                [word.text, word.pos_]
                for word in parsed_line] if d[0] != ' ']
        parsed_dialog.append(spacy_line)
        dialogues.append(parsed_dialog)
    del(raw_dialogues)
    return dialogues


def build_dataset(dialogues, min_length):
    input_data = []
    output_labels = [[] for _ in range(4)]
    spacy_parser = spacy.load('en')
    """
        For now, the data has four different layers:
            1. NOUN + PROPN + PRON
            2. NOUN + PROPN + PRON + VERB
            3. NOUN + PROPN + PRON + VERB + ADJ + ADV
            4. ALL
    """
    for idx, dialog in enumerate(dialogues):
        if idx % 1000 == 0:
            print_time_info(
                    "Parsed {}/{} dialogues".format(idx, len(dialogues)))
        attrs = []
        for attr_pair in dialog[0]:
            attrs.append(attr_pair[0])
            attrs.append(attr_pair[1])
        input_data.append(attrs)
        output_label = [[] for _ in range(4)]
        for w in dialog[1]:
            if w[0] in ["NAMETOKEN", "NEARTOKEN"]:
                w[1] = "NOUN"
            if w[1] in ['NOUN', 'PROPN', 'PRON']:
                output_label[0].append(w[0])
                output_label[1].append(w[0])
                output_label[2].append(w[0])
                output_label[3].append(w[0])
            elif w[1] == 'VERB':
                word = w[0]
                word = spacy_parser(word)[0].lemma_
                output_label[1].append(word)
                output_label[2].append(word)
                output_label[3].append(word)
            elif w[1] in ['ADJ', 'ADV']:
                output_label[2].append(w[0])
                output_label[3].append(w[0])
            else:
                if w[1] == "PUNCT":
                    pass
                else:
                    output_label[3].append(w[0])

        for idx in range(4):
            output_labels[idx].append(deepcopy(output_label[idx]))

    if min_length == -1:
        print_time_info(
                "No minimal length, data count: {}".format(len(dialogues)))
    else:
        print_time_info("Minimal length is {}".format(min_length))
        idxs = []
        for idx, sent in enumerate(input_data):
            if len(output_labels[3][idx]) > min_length:
                idxs.append(idx)
        input_data = [input_data[i] for i in idxs]
        output_labels = [
                [output_label[i] for i in idxs]
                for output_label in output_labels]
        print_time_info("Data count: {}".format(len(idxs)))
    return input_data, output_labels
