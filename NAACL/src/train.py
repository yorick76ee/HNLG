import argparse
import pickle
from model import NLG
from data_engine import DataEngine
from text_token import _UNK, _PAD, _BOS, _EOS
import torch
import torch.nn as nn
import numpy as np
import os
from utils import print_config, add_path
from model_utils import get_embeddings
from argument import define_arguments
from utils import get_time

_, args = define_arguments()

args = add_path(args)
print_config(args)

use_cuda = torch.cuda.is_available()
train_data_engine = DataEngine(
    data_dir=args.data_dir,
    save_path=args.train_data_file,
    vocab_path=args.vocab_file,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    regen=0,
    train=True
)

test_data_engine = DataEngine(
    data_dir=args.data_dir,
    save_path=args.valid_data_file,
    vocab_path=args.vocab_file,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    regen=0,
    train=False
)

vocab, rev_vocab, token_vocab, rev_token_vocab = \
        pickle.load(open(args.vocab_file, 'rb'))
en_vocab_size = len(token_vocab)
de_vocab_size = vocab_size = args.vocab_size + 4

if args.n_layers == 1:
    args.is_curriculum = 0

if args.use_embedding:
    embedding_dim = args.embedding_dim
    embeddings = get_embeddings(vocab, args.embeddings_dir, embedding_dim)
else:
    embeddings = None

model = NLG(
        n_decoders=args.n_layers,
        n_en_layers=args.n_en_layers,
        n_de_layers=args.n_de_layers,
        bidirectional=args.bidirectional,
        feed_last=args.feed_last,
        repeat_input=args.repeat_input,
        vocab_size=vocab_size,
        en_vocab_size=en_vocab_size,
        de_vocab_size=de_vocab_size,
        embedding_dim=args.embedding_dim,
        en_hidden_size=args.en_hidden_size,
        de_hidden_size=args.de_hidden_size,
        batch_size=args.batch_size,
        en_optimizer=args.en_optimizer,
        de_optimizer=args.de_optimizer,
        en_learning_rate=args.en_learning_rate,
        de_learning_rate=args.de_learning_rate,
        attn_method=args.attn_method,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        use_embedding=args.use_embedding,
        embeddings=embeddings,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        finetune_embedding=args.finetune_embedding,
        model_config=args
)


loss_weight = np.ones(args.vocab_size + 4)
loss_weight[_PAD] = 0.0
loss_weight[_EOS] = 1.0
loss_weight = torch.FloatTensor(loss_weight)
loss_weight = loss_weight.cuda() if use_cuda else loss_weight
# nn.NLLLoss(weight=loss_weight)
loss_func = nn.CrossEntropyLoss(weight=loss_weight)

if args.is_curriculum:
    for N in range(1, args.n_layers+1):
        model.train(
                epochs=args.epochs // args.n_layers,
                batch_size=args.batch_size,
                criterion=loss_func,
                verbose_epochs=args.verbose_epochs,
                verbose_batches=args.verbose_batches,
                valid_epochs=args.valid_epochs,
                valid_batches=args.valid_batches,
                save_epochs=args.save_epochs,
                inner_teacher_forcing_ratio=args.inner_teacher_forcing_ratio,
                inter_teacher_forcing_ratio=args.inter_teacher_forcing_ratio,
                inner_tf_decay_rate=args.inner_tf_decay_rate,
                inter_tf_decay_rate=args.inter_tf_decay_rate,
                max_norm=0.25,
                curriculum_layers=N)

else:
    model.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            criterion=loss_func,
            verbose_epochs=args.verbose_epochs,
            verbose_batches=args.verbose_batches,
            valid_epochs=args.valid_epochs,
            valid_batches=args.valid_batches,
            save_epochs=args.save_epochs,
            inner_teacher_forcing_ratio=args.inner_teacher_forcing_ratio,
            inter_teacher_forcing_ratio=args.inter_teacher_forcing_ratio,
            inner_tf_decay_rate=args.inner_tf_decay_rate,
            inter_tf_decay_rate=args.inter_tf_decay_rate,
            max_norm=args.max_norm,
            curriculum_layers=args.n_layers)
