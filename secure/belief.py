from dataclasses import dataclass

from torch import Tensor
from torch.distributions import Bernoulli
from logic_toolkit import Exp, AtomExp, DomainModel, Symbol, Reasoning

from secure.grounder import Grounder
from secure.utils import ObjectRecord

@dataclass
class Belief:

    def __init__(self,
                 idx: int,
                 exps: list[Exp],
                 records: list[ObjectRecord],
                 vocab: list[Symbol],
                 grounder: Grounder, 
                 priors: list[str, float],
                 semaware: bool = True,
                 base: float = 0.5,
                 custom_prior_weights: dict[AtomExp, tuple[float, float]] = None) -> None:
        """Belief state of the agent"""
        self.idx = idx
        self.exps = exps
        self.records = records
        self.vocab = vocab
        self.grounder = grounder
        self.priors = dict(priors)
        self.semaware = semaware
        self.base = base
        self.custom_prior_weights = custom_prior_weights

        # make sure that the grounder and its vocab are in sync
        self.grounder.add_symbols(self.vocab)

    def __str__(self):
        return f"Belief: {self.exps} Vocab: {self.vocab}"
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def entropy(self) -> Tensor:
        """The entropy of the belief state"""
        probs = Reasoning.compute_probs(self.exps, self.atoms, self.weights)
        return Tensor([Bernoulli(p).entropy() for p in probs.values()]).sum()
        
    @property
    def names(self) -> set[str]:
        return {record.name for record in self.records}
    
    @property
    def domain_size(self) -> int:
        return len(self.names)

    @property
    def prior_weights(self) -> dict[AtomExp, tuple[float, float]]:
        """Return the prior weights of the atoms in the belief state."""
        if self.custom_prior_weights is not None:
            return self.custom_prior_weights
        else:
            return {
                atom: (1.0-self.base,self.base) if atom.symbol.name not in self.priors \
                    else (1.0-self.priors[atom.symbol.name], self.priors[atom.symbol.name]) \
                        for atom in self.atoms 
            }

    @property
    def weights(self) -> dict[AtomExp, tuple[float, float]]:
        """Return the weights of the atoms in the belief state."""
        return self.grounder.pred_weights(self.records, self.vocab)
    
    @property
    def atoms(self) -> list[AtomExp]:
        return list(self.weights.keys())
    
    @property
    def map(self) -> DomainModel:
        """Return the MAP model of the belief state"""
        return Reasoning.MAP(self.exps, self.atoms, self.weights)[0]
    
    def update(self, exp: Exp) -> None:
        """Update the belief state with new exp evidence"""
        cnf = Reasoning.get_cnf([exp], set(self.records))
        # if CNF contains new symbols, add them to the vocab
        new_symbols = cnf.symbols - set(self.vocab)
        if new_symbols:
            self.grounder.add_symbols(new_symbols)
            self.vocab += new_symbols

        self.exps.append(cnf)
        
        probs = Reasoning.compute_probs(self.exps, self.atoms, self.prior_weights)

        self.grounder.update_records(self.idx, probs)

    def evi(self, exp: Exp) -> float:
        """Compute the probability of the expression, given the belief state"""
        cnf = Reasoning.get_cnf([exp], self.names)

        joint_prob = Reasoning.WMC([cnf]+self.exps, self.atoms, self.weights)
        marginal_prob = Reasoning.WMC(self.exps, self.atoms, self.weights)

        if not marginal_prob:
            return 0.0 # in case of zero marginal probability
        else:
            return min(1.0, joint_prob/marginal_prob) # guard for overflow
    
    
