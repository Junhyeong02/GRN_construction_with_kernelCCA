import torch
import torch.nn as nn
from torch_genometric.data import Data
from torch_geometric.nn.models import LabelPropagation

from torch_geometric.utils import subgraph

from typing import List
from dataclasses import dataclass

import networkx as nx

from scipy.stats import spearmanr

@dataclass
class GeneList:
    names:List[str]
    exps:List[float]

@dataclass
class DifferentialExpressionGeneList(GeneList):
    diff_exp_values:List[float]

@dataclass
class TranscriptionFactorGeneList(GeneList):
    pass

@dataclass
class GRN:
    node_name:List[str]
    node_attr:torch.Tensor

    edge_attr:torch.Tensor
    edge_index:torch.Tensor

    def get_node_indices_from_names(self, names:List[str])->List[int]:
        indices = []
        
        for i, node in enumerate(self.node_name):
            if node in names:
                indices.append(i)

        return indices
    
    def subgraph_from_names(self, names:List[str]):
        tgt = self.get_node_indices_from_names(names)

        edge_index, edge_attr = subgraph(torch.tensor(tgt), self.grn.edge_index, self.grn.edge_attr)
        node_attr = [self.node_attr[i] for i, b in enumerate(tgt) if b]

        return GRN(node_attr=node_attr, node_name=names, edge_attr=edge_attr, edge_index=edge_index)
    
class PropaNet:
    def __init__(self, grn:GRN, deg:DifferentialExpressionGeneList, tf:TranscriptionFactorGeneList, \
                 influence_round:int, num_propagate:int, alpha:float):
        super(PropaNet, self).__init__()
        
        self.grn = grn
        self.deg = deg
        self.tf = tf        

        self.influence = DEGInfluencer(influence_round)
        self.propagate = MaskedPropagation(num_propagate, alpha)
        
    def __call__(self):
        # step 1
        tgt_grn = self.grn.subgraph_from_names(set(self.deg.names + self.tf.names))

        # step 2
        tf_influence = self.influence(tgt_grn)

        # step 3
        score, mask = self.propagate(tgt_grn, tf_influence)

        return score, mask, tf_influence

class DEGInfluencer:
    def __init__(self, round):
        self.round = round

    def __call__(self, grn:GRN, x:torch.Tensor)->torch.Tensor:
        y = torch.zeros_like(x)

        for _ in range(self.round):
            y += self.influence(x)
        return y
    
    def influence():
        return 

class MaskedPropagation:
    def __init__(self, num_propagate, alpha):
        self.network_propagation = LabelPropagation(num_layers = num_propagate, alpha=alpha)
        self.metric = spearmanr

    def generate_mask(x):
        mask = torch.zeros_like(x)

        for _ in range(x.shape[0]):
            yield _
        # yield...

    def __call__(self, grn:GRN, x:torch.Tensor):
        score = -2
        mask_generater = self.generate_mask(x)
        mask = next(mask_generater)

        edge_index = grn.edge_index
        edge_weights = grn.edge_weight

        while True:    
            y = self.network_propagation(x, mask = mask, edge_index=edge_index, \
                                         edge_weightse=edge_weights)

            if self.metric(self.deg_values, y) < score:
                break

            mask = next(mask_generater)

        return mask, score 

