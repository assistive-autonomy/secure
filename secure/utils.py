import random
from typing import Iterable, Callable
from collections import defaultdict
from itertools import product
from dataclasses import dataclass, field

import torch
from torch import Tensor
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import f1_score, recall_score, precision_score

from PIL import Image
from logic_toolkit import (Entity,
                           DomainModel,
                           Symbol,
                           Semantics,
                           Denotation,
                           Reasoning,
                           AtomExp,
                           QuantExp,
                           Variable,
                           Exp,
                           Referent,
                           AndExp,
                           NegExp,
                           RefExp,
                           DenotationSet)
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

class ObjectRecord(Entity):
    """Object record to store the object information"""

    def __init__(self,
                name: str,
                bbox: Iterable[int],
                patch: Image = None,
                feats: Tensor = None,
                default_value: float = 0.5):
        super().__init__(name)
        self.bbox = bbox
        self.patch = patch
        self.feats = feats
        self.values: dict[Symbol, float] = defaultdict(lambda: default_value)
    
    @property
    def symbols(self) -> list[Symbol]:
        return sorted(self.values.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def width(self) -> int:
        return self.patch.size[0]
    
    @property
    def height(self) -> int:
        return self.patch.size[1]
    

def extract_records(env: RearrangementEnv,
                    model_name:str) -> list[ObjectRecord]:
    """create object records from the observation and props info."""
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    obs = env._compute_observation()['overhead_camera/rgb']

    records = []
    for name, props in env.props_info.items():
        bbox = props['bbox']
        patch = Image.fromarray(obs[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        feats = processor(patch, return_tensors="pt")
        feats = model(**feats).last_hidden_state.mean(dim=1).detach()
        record = ObjectRecord(name, bbox, patch, feats)
        # add symbols from the list
        records.append(record)
    
    # delete the processor and model for memory management
    del processor, model

    return records

def make_init_resolution_theory(model: DomainModel,refexp: RefExp) -> tuple[list[Exp],list[Symbol]]:
    """Make the initial domain theory about the resolution task"""

    def process_refexp(refexp: RefExp,
                       entities: Iterable[Entity]) -> tuple[Exp, set[Symbol]]:
        """Process the reference expression"""
        exp = Reasoning.make_base_exp(refexp)
        cnf = Reasoning.get_cnf([exp], set(entities))
        return cnf, cnf.symbols

    # all entities in the domain are objects
    obj_s = Symbol("object", 1)
    vocab = {obj_s}
    exps = [AtomExp.from_denotation(obj_s, Denotation(e)) for e in model.entities]

    # contraint to have a valid solution
    exp, vocab = process_refexp(refexp, model.entities)

    exps.append(exp)

    return exps, list(vocab)


def make_init_rearrangment_theory(model: DomainModel,
                    s_refexp: RefExp,
                    o_refexp: RefExp) -> tuple[list[Exp],list[Symbol]]:
    """Make the initial domain theory about the rearrangement task"""

    def process_refexp(refexp: RefExp,
                       entities: Iterable[Entity]) -> tuple[Exp, set[Symbol]]:
        """Process the reference expression"""
        exp = Reasoning.make_base_exp(refexp)
        cnf = Reasoning.get_cnf([exp], set(entities))
        return cnf, cnf.symbols

    # all entities in the domain are objects
    obj_s = Symbol("object", 1)
    vocab = {obj_s}
    exps = [AtomExp.from_denotation(obj_s, Denotation(e)) for e in model.entities]

    # contraint to have a valid solution
    s_exp, s_vocab = process_refexp(s_refexp, model.entities)
    o_exp, o_vocab = process_refexp(o_refexp, model.entities)

    exps += [s_exp, o_exp]
    vocab |= s_vocab | o_vocab

    # contrain to have at least one entity in referents
    s_exp, s_vocab = process_refexp(s_refexp.exists(), model.entities)
    o_exp, o_vocab = process_refexp(o_refexp.exists(), model.entities)

    exps += [s_exp, o_exp]
    vocab |= s_vocab | o_vocab

    # make sure that the subject and object referents are disjoint
    dis_exp= QuantExp("_every_q", Variable("x"), s_refexp.exp, 
                      NegExp(o_refexp.exp))
    dis_exp = Reasoning.get_cnf([dis_exp], set(model.entities))

    exps.append(dis_exp)

    return exps, list(vocab)


class Spatial:
    """Utilities to evaluate spatial relationships between objects"""

    surface2sym = {
        "to the left of": Symbol("left", 2),
        "to the right of": Symbol("right", 2),
        "in front of": Symbol("front", 2),
        "behind": Symbol("behind", 2),
    }
    sym2surface = {v: k for k, v in surface2sym.items()}

    # Spatial relationships predicates using PASCAL VOC format bounding boxes

    
    def is_left(s_bbox: Iterable[int],o_bbox:Iterable[int]) -> bool:
        """ s[xmax] < o[xmin] """
        return s_bbox[2] < o_bbox[0]

    def is_right(s_bbox: Iterable[int], o_bbox:Iterable[int]) -> bool:
        """ s[xmin] > o[xmax] """
        return s_bbox[0] > o_bbox[2]
    
    def is_behind(s_bbox: Iterable[int], o_bbox:Iterable[int]) -> bool:
        """ s[ymax] < o[ymin] """
        return s_bbox[3] < o_bbox[1]

    def is_front(s_bbox: Iterable[int], o_bbox:Iterable[int]) -> bool:
        """ s[ymin] > o[ymax] """
        return s_bbox[1] > o_bbox[3]

    def get_rel_from_surface(surface: str) -> Callable[[Iterable[int], Iterable[int], float], bool]:
        """Get the relationship function from the surface"""
        match surface:

            case "to the left of":
                return Spatial.is_left
            
            case "to the right of":
                return Spatial.is_right

            case "in front of":
                return Spatial.is_front
            
            case "behind":
                return Spatial.is_behind
                        
            case _:
                raise NotImplementedError(f"{surface} not supported")
            
    def get_rel_from_symbol(symbol: Symbol) -> Callable[[Iterable[int], Iterable[int], float], bool]:
        """Get the relationship function from the symbol"""
        return Spatial.get_rel_from_surface(Spatial.sym2surface[symbol])
            
    @staticmethod
    def sample() -> tuple[str,Callable[[Iterable[int], Iterable[int], float], bool]]:
        """Sample a random spatial relationship"""
        rel = random.choice(list(Spatial.surface2sym.keys()))
        fun = Spatial.get_rel_from_surface(rel)

        return rel, fun
    
    @staticmethod
    def eval(surface_or_sym:str|Symbol,s_bbox: np.ndarray,o_bbox) -> bool:
        """evaluate the spatial relationship between two objects"""
        if surface_or_sym in Spatial.surface2sym:
            rel_func = Spatial.get_rel_from_surface(surface_or_sym)
        else:
            rel_func = Spatial.get_rel_from_symbol(surface_or_sym)

        return rel_func(s_bbox, o_bbox)
    
    @staticmethod
    def get_region_poses(props: list[int],
                         surface:str|Symbol,
                         env:RearrangementEnv, 
                         camera:str,
                         safety_dist_pixels: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Get the region poses based on the spatial relationship"""
        if isinstance(surface, Symbol):
            surface = Spatial.sym2surface[surface]

        bboxs = [env.props_info[prop]['bbox'] for prop in props]

        w_min_pose = env._cfg.task.initializers.workspace.min_pose
        w_max_pose = env._cfg.task.initializers.workspace.max_pose

        max_x, max_y = env.world_2_pixel(camera, w_min_pose)
        min_x, min_y  = env.world_2_pixel(camera, w_max_pose)
            
        for bbox in bboxs:

            match surface:
                
                # x_min y_min x_max y_max
                case "to the right of": 
                    min_x = max(min_x, bbox[2]) + safety_dist_pixels
                case "to the left of":
                    max_x = min(max_x, bbox[0]) - safety_dist_pixels
                case "behind":
                    max_y = min(max_y, bbox[1]) - safety_dist_pixels
                case "in front of":
                    min_y = max(min_y, bbox[3]) + safety_dist_pixels

        r_min_coord = np.array([min_x, min_y])
        r_max_coord = np.array([max_x, max_y])

        # get min-max poses
        r_max_pose = env.pixel_2_world(camera, r_min_coord)
        r_min_pose = env.pixel_2_world(camera, r_max_coord)

        return r_min_pose, r_max_pose

def make_domain_model(props_info: dict) -> DomainModel:
    """Make the domain model from the object records and props info"""
    
    extensions = defaultdict(list)
    prop2entity = {prop : Entity(str(prop)) for prop in props_info}

    for prop, entity in prop2entity.items():

        den = Denotation(entity)
        labels = list(props_info[prop]['labels'].data.values())
        for label in labels:
            symbol = Symbol(label, 1)
            extensions[symbol].append(den)

    # add spatial relationships to the domain
    for rel_str, sym in Spatial.surface2sym.items():
        for s_prop, o_prop in product(prop2entity, prop2entity):
            s_bbox = props_info[s_prop]['bbox']
            o_bbox = props_info[o_prop]['bbox']
            if Spatial.eval(rel_str, s_bbox, o_bbox):
                den = Denotation((prop2entity[s_prop], prop2entity[o_prop]))
                extensions[sym].append(den)

    entities = set(list(prop2entity.values()))
    extensions = {k: DenotationSet(v) for k, v in extensions.items()}

    return DomainModel(entities, extensions)

def task_f1_score(pred_model: DomainModel,
             target_model: DomainModel, 
             refexps: list[RefExp], include_pr=False)->float:
    """Compute the f1 score for the list of referential expressions"""
    target_entities = target_model.entities
    pred_entities = pred_model.entities

    preds, targets = [], []
    for refexp in refexps:
        pred = Semantics.eval_refexp(refexp, pred_model).entities
        target = Semantics.eval_refexp(refexp, target_model).entities
        pred = Tensor([1 if entity in pred else 0 for entity in pred_entities])
        target = Tensor([1 if entity in target else 0 for entity in target_entities])
        preds.append(pred)
        targets.append(target)

    preds = torch.stack(preds, dim=1)
    targets = torch.stack(targets, dim=1)

    f1 = f1_score(targets, preds, average='micro')

    if include_pr:
        recall = recall_score(targets, preds, average='micro')
        precision = precision_score(targets, preds, average='micro')
        return f1, recall, precision
    else:
        return f1


def process_correction(d_refexp: RefExp,
                       t_refexp: RefExp, 
                       ref: Referent, 
                       entities: Iterable[Entity]) -> Exp:
    """Create logic formula from the correction"""
    
    t_refexp = t_refexp.exists() # make sure that the exists refexp is used
    d_exp = Reasoning.process_refexp(d_refexp, ref, entities) # designation exp
    t_exp = Reasoning.process_refexp(t_refexp, ref, entities) # truth-condition exp

    return AndExp(d_exp, NegExp(t_exp))