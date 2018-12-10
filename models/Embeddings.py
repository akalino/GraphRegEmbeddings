import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out)
        return log_probs

class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out)
        return log_probs

class GCBOW(nn.Module):
    
    def __init__(self, vocab_size, embeding_dim, graph_weights, reg_factor):
        super(GCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.reg_factor = reg_factor
        self.graph_weights = torch.sparse_coo_tensor(ind, val, dtype=torch.float32, requires_grad=False)

    def forward(self, inputs)
        embeds = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
        out = self.linear(embeds)
        log_probs = (1 - self.reg_factor) * F.log_softmax(out) \
                - self.reg_factor * torch.sum(self.graph_weights, dim=0)
        return log_probs

class GSkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, reg_factor):
        super(GSkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.reg_factor = reg_factor
        self.graph_weights = torch.sparse_coo_tensor(ind, val, dtype=torch.float32, requires_grad=False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = (1 - self.reg_factor) * F.log_softmax(out) \
                - self.reg_factor * torch.sum(self.graph_weights, dim=0)
        return log_probs

