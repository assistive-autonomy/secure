from typing import Iterable

from torch import Tensor
from logic_toolkit import GRM, RefExp, Entity, Semantics, Reasoning, AndExp

from secure.task import Task, RearrangementTask
from secure.belief import Belief
from secure.messages import Message, Query, CompleteResponse, Correction, Complete, Clarification


def cost(query: Query,
         symbol_cost:float,
         point_cost:float,
         entities: Iterable[Entity]) -> Tensor:
    """Query cost"""

    def N_point(refexp: RefExp, entities: Iterable[Entity]) -> int:
        if GRM.A.matches(refexp.name):
            return 1
        elif GRM.EVERY.matches(refexp.name):
            return (len(entities)-1)/2
        elif GRM.UNIQUE.matches(refexp.name):
            parts =  refexp.name.split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
        else:
            raise NotImplementedError("Not implemented yet.")

    n_point = N_point(query.refexp_lf, entities)
    n_symbols = len(query.refexp_lf.symbols)

    return Tensor([point_cost*n_point + symbol_cost*n_symbols])


def reward(action: Message,
           response: Message,
           symbol_cost: float,
           point_cost:float,
           entities: Iterable[Entity]) -> Tensor:
    """reward function after action and response"""

    if isinstance(action, Query) or isinstance(action, Clarification):
        return -cost(action, symbol_cost, point_cost, entities)
    elif isinstance(response, CompleteResponse):
        return Tensor([1.0])
    elif isinstance(response, Correction):
        return Tensor([-1.0])
    else:
        raise ValueError(f"not supported response: {response}")

def expected_reward(action: Message,
                    belief: Belief,
                    task: Task,
                    symbol_cost: float,
                    point_cost: float,
                    exists: bool,
                    entities: Iterable[Entity]) -> Tensor:
    """expected reward, given belief state and action"""
    if isinstance(action, Query):
        return -cost(action, symbol_cost, point_cost, entities)
    else:
        """Return the MAP probability of the belief state"""
        # get referencs from the task
        pred_model = belief.map
        s_ref = Semantics.eval_refexp(task.s_refexp_lf,
                                      pred_model,
                                      exists)
        exp = Reasoning.process_refexp(task.s_refexp_lf,
                                         s_ref,
                                         entities,
                                         exists)
        
        if isinstance(task, RearrangementTask):

            o_ref = Semantics.eval_refexp(task.o_refexp_lf,
                                        pred_model,
                                        exists)    
            o_exp = Reasoning.process_refexp(task.o_refexp_lf,
                                            o_ref,
                                            entities,
                                            exists)
        
            exp = AndExp(exp, o_exp)
            
        prob = Tensor([belief.evi(exp)])
        
        return 2*prob - 1 # normalize to [-1, 1]


