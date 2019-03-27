"""
Data pre-processing file
"""

import pandas as pd
import numpy as np
from torchtext import data, vocab
import torch
import spacy
import os
from random import randint

spacy_en = spacy.load('en')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = os.path.join('dataset')


class citeulike:
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """

    def __init__(self, batch_size=100):
        print('Device: ' + str(device))
        self.user = data.Field(sequential=False, use_vocab=False)
        self.doc_title = data.Field(sequential=True, lower=True, include_lengths=True)
        self.ratings = data.Field(sequential=False, use_vocab=False)
        # self.doc_abstract = data.Field(sequential=True, tokenize=tokenizer, lower=True)

        self.train_set, self.validation_set = data.TabularDataset.splits(
            path=dataset,
            train='train_data.csv',
            validation='val_data.csv',
            format='csv',
            fields=[
                ('index', None),
                ('user', self.user),
                ('doc_id', None),
                ('ratings', self.ratings),
                ('doc_title', self.doc_title),
                # ('doc_abstract', self.doc_abstract)
            ],
            skip_header=True,
        )

        self.train_iter, self.validation_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set),
            batch_size=batch_size,
            shuffle=True,
            device=device,
            sort_key=lambda x: len(x.doc_title),
            sort_within_batch=True,
            repeat=True)

        self.user.build_vocab(self.train_set)
        self.ratings.build_vocab(self.train_set)
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
        # self.doc_abstract.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))
        self.doc_title.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.en.vec', url=url))


def to_csv_citeulike(total=0):
    docs = pd.read_csv(os.path.join(dataset, 'raw-data.csv'), usecols=['doc.id', 'raw.title', 'raw.abstract'],
                       dtype={'doc.id': np.int32, 'raw.title': str, 'raw.abstract': str}, header=0, sep=',')
    users = pd.read_csv(os.path.join(dataset, 'user-info.csv'), usecols=['user.id', 'doc.id', 'rating'], header=0,
                        sep=',')
    docs.set_index('doc.id', inplace=True)
    titles, abstracts, users_list, ratings, docs_list = [], [], [], [], []
    val_titles, val_abstracts, val_users_list, val_ratings, val_docs_list = [], [], [], [], []
    cnt = 0
    max_user = users.iloc[-1]['user.id'] if total is 0 else users.iloc[total]['user.id']
    for index, row in users.iterrows():
        cnt += 1
        if total is not 0:
            if cnt == total:
                break
        if cnt % 5 == 0:
            val_titles.append(docs.loc[row['doc.id']]['raw.title'])
            val_abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
            val_users_list.append(row['user.id'] - 1)
            val_ratings.append(1)
            val_docs_list.append(row['doc.id'])
            continue

        titles.append(docs.loc[row['doc.id']]['raw.title'])
        abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
        users_list.append(row['user.id'] - 1)
        ratings.append(row['rating'])
        docs_list.append(row['doc.id'])
        titles.append(docs.loc[row['doc.id']]['raw.title'])
        abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
        users_list.append(randint(0, max_user))
        ratings.append(0)
        docs_list.append(row['doc.id'])

    d = {'user.id': users_list, 'doc.id': docs_list, 'rating': ratings, 'raw.title': titles,
         'raw.abstract': abstracts}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(dataset, 'train_data.csv'))

    d = {'user.id': val_users_list, 'doc.id': val_docs_list, 'rating': val_ratings, 'raw.title': val_titles,
         'raw.abstract': val_abstracts}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(dataset, 'val_data.csv'))


def tokenizer(text):
    """
    Tokenizer function
    :param text:
    :return: tokens
    """
    stop_words = {'(', ')', '/', 'm', 'w', '-', ' ', '.', '\t'}
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    tokens = list(filter(lambda token: token not in stop_words, tokens))
    return tokens