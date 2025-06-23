import os
from copy import deepcopy
import random

import inflect
import torch
from torch import Tensor
import torch.nn as nn
from logic_toolkit import (Semantics, Reasoning, Entity, RefExpParser, Denotation,
                           RefExp, Referent, DenotationSet)

from secure.messages import Message, Query, ActExecution, Clarification
from secure.task import Task, RearrangementTask, InteractiveResolutionTask
from secure.belief import Belief
from secure.reward import expected_reward

class Learner:
    def __init__(self,
                 name:str,
                 learning_rate:float,
                 epsilon:float,
                 gamma:float,
                 init_weights: list[float],
                 point_cost:float,
                 symbol_cost:float,
                 update_freq:int,
                 semaware: bool,
                 dialogue: bool,
                 load_path: str,
                 save_path: str,
                 load: bool,
                 save: bool,
                 task: Task = None) -> None:
        
        assert len(init_weights) == 2

        self.name = name
        # SARSA learning params
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_weights = init_weights
        self.update_freq = update_freq

        # cost params
        self.point_cost = point_cost
        self.symbol_cost = symbol_cost

        # semantic analysis 
        self.semaware = semaware
        self.syntax = inflect.engine()

        # dialogue mode
        self.dialogue = dialogue

        # load and save params
        self.load_path = load_path
        self.save_path = save_path
        self.load = load
        self.save = save

        self.task = task
        self.asked_queries = [] # list of queries asked by the agent

        if self.load and self.load_path:
            self.params = torch.load(self.load_path)
        else:
            self.params = nn.Linear(2,1, bias=False)
            self.params.weight.data = Tensor([self.init_weights])

        self.params_target = deepcopy(self.params)

        self.optimizer = torch.optim.Adam(self.params.parameters(),
                                          lr=self.lr)
        self.iter_num = 0


    def set_task(self, task: Task) -> None:
        self.task = task

    @property
    def actions(self) -> list[Message]:
        """Return the list of possible actions for the agent."""
        if self.task is None:
            return []
        elif isinstance(self.task, RearrangementTask) and not self.dialogue:
            return [ActExecution(agent=self.name)]
        elif isinstance(self.task, RearrangementTask):
            return list(set([
                ActExecution(agent=self.name),
                Query(self.task.s_refexp, self.task.s_refexp_lf, agent=self.name),
                Query(self.task.s_refexp_exists, self.task.s_refexp_exists_lf, agent=self.name),
                Query(self.task.s_refexp_every, self.task.s_refexp_every_lf, agent=self.name),
                Query(self.task.o_refexp, self.task.o_refexp_lf, agent=self.name),
                Query(self.task.o_refexp_exists, self.task.o_refexp_exists_lf, agent=self.name),
                Query(self.task.o_refexp_every, self.task.o_refexp_every_lf, agent=self.name),
            ]))
        elif isinstance(self.task, InteractiveResolutionTask):
            ## Action space for Interactive Resolution Task include Act, Queries and Clarifications

            act = ActExecution(agent=self.name)

            queries = [Query(self.task.s_refexp, self.task.s_refexp_lf, agent=self.name),
                       Query(self.task.s_refexp_exists, self.task.s_refexp_exists_lf, agent=self.name),
                       Query(self.task.s_refexp_every, self.task.s_refexp_every_lf, agent=self.name)]
            
            # get symbols of s_refexp_lf
            clars = []
            symbols = self.task.s_refexp_lf.symbols
            parser = RefExpParser()
            for symbol in symbols:
                refexp_str = self.syntax.a(symbol.name)
                refexp_lf_str = f"< _a_q x. {symbol.name}(x)>"
                refexp_lf = parser(refexp_lf_str)
                clar = Clarification(refexp_str, refexp_lf, agent=self.name)
                clars.append(clar)
            return [act] + queries + clars
        else:
            raise ValueError(f"not supported action space for task: {self.task}")


    def compute_exp_ent(self,
                    belief: Belief,
                    refexp: RefExp,
                    entities: set[Entity],
                    use_exists:bool=False) -> Tensor:
        """Compute the expected entropy """

        if use_exists:
            refexp = refexp.exists()

        ents = []
        probs = []
        dens = list(Semantics.make_referent(refexp, entities, not self.semaware)._data)[0]
        for den in dens:
            ref = Referent(DenotationSet(den))
            exp = Reasoning.process_refexp(refexp, ref, entities, not self.semaware)
            prob_exp = Tensor([belief.evi(exp)])
            if not prob_exp:
                """skip the case where the probability is zero."""
                continue
            pos_belief = deepcopy(belief)
            pos_belief.update(exp)

            ent = pos_belief.entropy
                    
            probs.append(prob_exp)
            ents.append(ent)

        if not probs:
            # no valid answer yields to no information gain
            return Tensor([belief.entropy])
        else:
            # vectorize & normalize
            probs = torch.stack(probs).flatten().softmax(dim=0)
            ents = torch.stack(ents).flatten()

            return torch.dot(probs, ents)

    def q_values(self,
                 belief: Belief,
                 params: nn.Linear) -> tuple[dict[Message, Tensor]]:

    
        curr_ent = belief.entropy
        entities = {Entity(str(record.name)) for record in belief.records}
        qs, gains, exp_rewards = {}, {}, {}

        # update action space so that for each entity we a clarrification
        actions = [action for action in self.actions if not isinstance(action, Clarification)]
        clf_action_templates = [action for action in self.actions if isinstance(action, Clarification)]

        for clf_action in clf_action_templates:
            for entity in entities:
                referent = Referent(DenotationSet(Denotation(entity)))
                action = deepcopy(clf_action)
                action.referent = referent
                action.text = clf_action.text.replace("None", str(action.referent))

                actions.append(action)
        for action in actions:

            reward = expected_reward(action,
                                     belief,
                                     self.task,
                                     self.symbol_cost,
                                     self.point_cost,
                                     True, 
                                     entities)
            
            if isinstance(action, ActExecution) and isinstance(self.task, RearrangementTask):
                    
                s_exp_ent = self.compute_exp_ent(belief, self.task.s_refexp_lf, entities, use_exists=not self.semaware)
                o_exp_ent = self.compute_exp_ent(belief, self.task.o_refexp_lf, entities, use_exists=not self.semaware)
                # jointly normalize probs
                exp_ent = (s_exp_ent + o_exp_ent)/2

            elif isinstance(action, ActExecution) and isinstance(self.task, InteractiveResolutionTask):
                exp_ent = self.compute_exp_ent(belief, self.task.s_refexp_lf, entities, use_exists=not self.semaware)

            elif isinstance(action, Query):
                exp_ent = self.compute_exp_ent(belief, action.refexp_lf, entities, use_exists=not self.semaware)
            else:
                # Clarification so do compute for each entity
                exp_ent = self.compute_exp_ent(belief, action.refexp_lf, referent.entities, use_exists=not self.semaware)
            
            gain = curr_ent - exp_ent
        
            feats = Tensor([gain, reward])
            qs[action] = params(feats)
            gains[action] = gain
            exp_rewards[action] = reward

        return qs, gains, exp_rewards

    
    def greedy_action(self, qs: dict[Message, Tensor]) -> Message:
        """choose a greedy (most valuable) action"""
        qs = {k: v.item() for k, v in qs.items() if k not in self.asked_queries or isinstance(k, ActExecution)}
        action = max(qs, key=qs.get)
        if not isinstance(action, ActExecution):
            self.asked_queries.append(action)
        return action

    def random_action(self, qs: dict[Message, Tensor]) -> Message:
        """random action"""
        qs = {k: v.item() for k, v in qs.items() if k not in self.asked_queries or isinstance(k, ActExecution)}
        action = random.choice(list(qs.keys()))
        if not isinstance(action, ActExecution):
            self.asked_queries.append(action)
        return action

    def choose_action(self, qs: dict[Message, Tensor]) -> Message:
        """Choose action in epsilon-greedy manner"""
        if random.random() > self.epsilon:
            return self.greedy_action(qs)
        else:
            return self.random_action(qs)
        
    def train(self,
              q_value: Tensor,
              action: Message,
              reward: Tensor,
              next_belief: Belief) -> None:
        """Train preference function"""
        target_qs = self.q_values(next_belief, self.params_target)[0]
        target_action = self.choose_action(target_qs)
        target_q = target_qs[target_action]

        if not isinstance(action, ActExecution):
            target = self.gamma * target_q + reward
        else:
            target = reward
            return
        
        loss = (q_value - target).pow(2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iter_num += 1
        if self.iter_num % self.update_freq == 0:
            self.params_target.load_state_dict(self.params.state_dict())

    def save_params(self, filename:str) -> None:
        """Save the parameters of the agent."""
        filename = os.path.join(self.save_path, filename)+".pt"
        torch.save(self.params, filename)