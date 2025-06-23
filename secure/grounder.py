import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Bernoulli

from logic_toolkit import Symbol, AtomExp, Denotation, Entity
from secure.utils import ObjectRecord

class Grounder:
    def __init__(self,
                 memory: dict[int, list[ObjectRecord]],
                 vocab: list[Symbol],
                 feat_size: int, 
                 threshold: float,
                 ):
        """Entity Grounding model based on Multilabel Prototype Network"""
        assert 0.0 < threshold < 1.0

        self.memory = memory
        self.vocab = vocab
        self.feat_size = feat_size
        self.threshold = threshold
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def records(self):
        return [r for rs in self.memory.values() for r in rs]
    

    def add_symbol(self, symbol: Symbol, init_value: float = 0.5) -> None:
        assert 0.0 < init_value < 1.0

        if symbol not in self.vocab:
            self.vocab.append(symbol)
            # extend records with new symbol
            for record in self.records:
                record.values[symbol] = init_value

    def add_symbols(self, symbols: list[Symbol], init_value: float = 0.5) -> None:
        for symbol in symbols:
            self.add_symbol(symbol, init_value)

    def add_records(self, idx: int, records: list[ObjectRecord]) -> None:
        """add records without dublicates"""
        self.memory[idx] = records

    def update_records(self, idx: int, probs: dict[AtomExp, float]) -> None:
        assert idx in self.memory, "Unknown idx"
        """update the records with the new weights"""
        for record in self.memory[idx]:
            for atom, prob in probs.items():
                if int(atom.terms[0].name) == record.name:
                    record.values[atom.symbol] = prob

    def get_support(
        self, symbol: Symbol
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """get possitive and negative support for a given symbol"""
        assert symbol in self.vocab

        pos_feats, neg_feats, pos_values, neg_values = [], [], [], []

        for record in self.records:
            value = record.values[symbol]
            entropy = Bernoulli(value).entropy()
            
            if value > 0.5 and entropy < self.threshold:
                pos_feats.append(record.feats)
                pos_values.append(value)
            elif value < 0.5 and entropy < self.threshold:
                neg_feats.append(record.feats)
                neg_values.append(value)

        # In case no examples are stored, default to random record
        if len(pos_feats) == 0 or len(pos_values) == 0:
            pos_record = max(self.records, key=lambda r: r.values[symbol])
            pos_feats = pos_record.feats.view(1, -1)
            pos_values = Tensor([pos_record.values[symbol]]).view(1, -1)
        else:
            pos_feats = torch.cat(pos_feats).view(len(pos_feats), -1)
            pos_values = Tensor([pos_values]).view(len(pos_values), -1)

        if len(neg_feats) == 0 or len(neg_values) == 0:
            neg_record = min(self.records, key=lambda r: r.values[symbol])
            neg_feats = neg_record.feats.view(1, -1)
            neg_values = Tensor([neg_record.values[symbol]]).view(1, -1)
        else:
            neg_feats = torch.cat(neg_feats).view(len(neg_feats), -1)
            neg_values = Tensor([neg_values]).view(len(neg_values), -1)

        return pos_feats, pos_values, neg_feats, neg_values


    def pred_values(self,
                   record: ObjectRecord,
                   symbols: list[Symbol]) -> dict[Symbol: float]:
        """Compute the concept vector for the query record"""
        assert all(symbol in self.vocab for symbol in symbols), "Unknown symbol in symbols"
        # extract features for the query patch
        query = record.feats
        
        values = {}
        for symbol in symbols:

            pos_sup, pos_labels, neg_sup, neg_labels = self.get_support(symbol)

            pos_prototype = torch.mean(pos_labels * pos_sup, dim=0).flatten()
            neg_prototype = torch.mean((1 - neg_labels) * neg_sup, dim=0).flatten()

            cos_sim_neg = F.cosine_similarity(query, neg_prototype)
            cos_sim_pos = F.cosine_similarity(query, pos_prototype)

            logit = cos_sim_pos - cos_sim_neg
            
            values[symbol] = float(torch.sigmoid(logit))

        return values
    
    def pred_weights(self,
                     records: list[ObjectRecord],
                     symbols: list[Symbol]) -> dict[AtomExp, tuple[float, float]]:
        """predict the weigths to be used to update the belief state of the agent."""
        weights = {}
        for record in records:
            values = self.pred_values(record, symbols)
            den = Denotation(Entity(str(record.name)))
            for symbol, value in values.items():
                atom = AtomExp.from_denotation(symbol, den)
                weights[atom] = (1 - value, value)
        return weights