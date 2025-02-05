import os
import torch.optim

from data_loader import TextDS
from models import FastText
from plotting import *
from training import train_model

num_epochs = 10
num_hidden = 128

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TextDS(os.path.join('data', 'unlabelled_movie_reviews.csv'), dev=dev)

model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.token_to_id)+2).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters()training accuracy: 0.7016000151634216
# validation accuracy: 0.7015999555587769
# testing accuracy: 0.7032999992370605, lr=0.1, momentum=0.9)
losses, accuracies = train_model(dataset, model, optimizer, num_epochs)
torch.save(model, os.path.join('saved_models', 'word_embeddings.pth'))

print_accuracies(accuracies)
plot_losses(losses)
