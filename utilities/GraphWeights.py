import re
import torch
import numpy as np


def relation_mapping(_path):
    with open(_path, 'r') as f:
        rel_dict = {}
        rels = f.readlines()
        for l in rels:
            m = re.sub('\t', ' ', l)
            m = re.sub('\n', '', m).split()
            if len(m) > 1:
                rel_dict[m[0]] = int(m[1])
    return rel_dict


def init_graph(_path):
    with open(_path, 'r') as g:
        ents = g.readline()
        reg = np.zeros((int(ents), int(ents)))
    return reg


def add_weights(_path, _graph):
    with open(_path, 'r') as f:
        for l in f.readlines():
            m = re.sub('\t', ' ', l)
            m = re.sub('\n', '', m).split()
            m = [int(k) for k in m]
            if len(m) > 1 and m[2] == 7:
                _graph[m[0], m[1]] = 1.0
            elif len(m) > 1 and m[2] in [0, 8]:
                _graph[m[0], m[1]] = 0.5
    return _graph


if __name__ == "__main__":
    rel_map = relation_mapping('/home/ak/Projects/GraphRegEmbeddings/WN11/relation2id.txt')
    graph_reg = init_graph('/home/ak/Projects/GraphRegEmbeddings/WN11/entity2id.txt')
    p1 = '/home/ak/Projects/GraphRegEmbeddings/WN11/train2id.txt'
    p2 = '/home/ak/Projects/GraphRegEmbeddings/WN11/test2id.txt'
    p3 = '/home/ak/Projects/GraphRegEmbeddings/WN11/valid2id.txt'
    graph_reg = add_weights(p1, graph_reg)
    graph_reg = add_weights(p2, graph_reg)
    graph_reg = add_weights(p3, graph_reg)
    out = torch.from_numpy(graph_reg)
    out.requires_grad = False
    torch.save(out, 'GraphWN11.pt')
