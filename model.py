"""
Model file
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn

max_rating = 5.0
min_rating = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CiteULikeModel(nn.Module):
    """
    Colaboratie filtering model for article-author paring
    """

    def __init__(self, text_vectors, user_field, user_dim=10, l1=50, l2=50, p1=0.3, p2=0.3, p3=0.3):
        """
        :param text_vectors: Field for the article texts
        :type text_vectors: torch.Tensor
        :param user_field: Field for authors
        :type user_field: torchtext.data.Field
        :param user_dim: Dimensionality of the author embedding
        :param l1: Number of hidden units in the 1st layer
        :param l2: Number of hidden units in the 2nd layer
        :param p1: Dropout probability for the 1st layer
        :param p2: Dropout probability for the 2nd layer
        :param p3: Dropout probability in the output layer
        """
        super(CiteULikeModel, self).__init__()

        num_embeddings = text_vectors.size()[0]
        embedding_dim = text_vectors.size()[1]

        self.article_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.article_embeddings.weight.data.copy_(text_vectors)

        num_author = len(user_field.vocab.freqs)
        self.user_embedding = nn.Embedding(num_author, user_dim)
        self.user_embedding.weight.data.uniform_(0, 200)

        self.l_1 = nn.Sequential(
            nn.Dropout(p1),
            nn.Linear(in_features=(embedding_dim + user_dim),
                      out_features=l1,
                      bias=True),
            nn.ReLU(),
        )

        self.l_2 = nn.Sequential(
            nn.Dropout(p2),
            nn.Linear(in_features=l1,
                      out_features=l2,
                      bias=True),
            nn.ReLU(),
        )

        self.l_out = nn.Sequential(
            nn.Dropout(p3),
            nn.Linear(in_features=l2,
                      out_features=1,
                      bias=True),
        )

    def forward(self, x):
        user = self.user_embedding(x.user)
        text = torch.mean(self.article_embeddings(x.text), dim=0)
        x = torch.cat((user, text), 1)

        x = self.l_1(x)
        x = self.l_2(x)

        out = torch.sigmoid(self.l_out(x))
        return out


class LstmNet(nn.Module):
    def __init__(self, article_field, user_field, user_dim=50, hidden_dim=50, lstm_layers=5):
        super(LstmNet, self).__init__()
        article_vectors = article_field.vocab.vectors
        num_embeddings = article_vectors.size()[0]
        embedding_dim = article_vectors.size()[1]

        self.article_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.article_embeddings.weight.data.copy_(article_vectors)

        num_author = len(user_field.vocab.freqs)
        self.author_embedding = nn.Embedding(num_author, user_dim)
        self.author_embedding.weight.data.uniform_(0, 0.01)

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers)

        self.linear = nn.Sequential(
            nn.Linear(in_features=(user_dim + hidden_dim),
                      out_features=1,
                      bias=True),
            nn.ReLU(),
        )

    def forward(self, x, lengths):
        user = self.author_embedding(x.user)
        text = self.article_embeddings(x.doc_title)

        ## Packing and padding
        packed = rnn.pack_padded_sequence(text, lengths)
        lstm_out, (lstm_hidden, lstm_state) = self.lstm(packed)

        x = (user * lstm_state[-1]).sum(1)

        out = torch.sigmoid(x)

        return out
