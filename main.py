"""
Main
"""
import torch
from torch import optim, nn
from model import LstmNet
from data import citeulike
from train import train_with_negative_sampling

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

citeulike = citeulike(batch_size=200)

train_iter = citeulike.train_iter
validation_iter = citeulike.validation_iter

user_field = citeulike.user
title_field = citeulike.doc_title

net = LstmNet(article_field=title_field, user_field=user_field).to(device)
opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()
train_with_negative_sampling(train_iter=train_iter, val_iter=validation_iter, net=net,
                             optimizer=opt, criterion=criterion, num_epochs=50)
