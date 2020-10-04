import torch.nn as nn
import torch

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        # self.W = nn.Linear(embedding_dim, num_classes)
        self.multi_layers = nn.Sequential(
            nn.Linear(embedding_dim,512),
            nn.Dropout(p=0.5),
            nn.Sigmoid(),
            nn.Linear(512,2048),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        self.W = nn.Linear(2048,num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """

        # TODO perform embed, aggregate, and linear, then return the predicted class probabilities.
        x = self.embeddings(x)
        # multi-layer average
        multi_out = self.multi_layers(torch.mean(x, dim=1))
        out = self.W(multi_out)
        # # original mean imple
        # out = self.W(torch.mean(x, dim=1))
        # # bad max imple
        # maxout, index = torch.max(x, axis=1)
        # out = self.W(maxout)
        return out

