import gc
from copy import deepcopy
from typing import Callable, Iterable
from dataclasses import dataclass
import random
import inflect
from logic_toolkit import Semantics
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv, DEFAULT_CONFIG

from secure.utils import Spatial, extract_records, make_domain_model, ObjectRecord
from secure.messages import ActExecution, Query
from secure.generator import Generator

QUANT = ["unique", "every", "exists"]

## TRAINING CONFIG
TRAIN_CONFIG = deepcopy(DEFAULT_CONFIG)
# PROP CONFIG
TRAIN_CONFIG['arena']['props'] = {
    "min_objects": 4,
    "max_objects": 9,
    "min_object_size": 0.015,
    "max_object_size": 0.016,
    "sample_size": True,
    "sample_colour": True,
    "color_noise": 0.2,
    "shapes": ["cube"],
    "colours": ["green", "blue", "red"],
    "textures": ["plain"],
}
# PHYSICS CONFIG
TRAIN_CONFIG["physics_dt"] = 0.001
TRAIN_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["gains"]["position"]["kp"] = 300
TRAIN_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["gains"]["orientation"]["kp"] = 800
TRAIN_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["convergence"]["position_threshold"] = 5e-3
TRAIN_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["convergence"]["orientation_threshold"] = 0.17

## EVALUATION CONFIG
EVAL_CONFIG = deepcopy(DEFAULT_CONFIG)
# PROP CONFIG
EVAL_CONFIG['arena']['props'] = {
    "min_objects": 4,
    "max_objects": 9,
    "min_object_size": 0.015,
    "max_object_size": 0.016,
    "sample_size": True,
    "sample_colour": True,
    "color_noise": 0.2,
    "shapes": ["cube", "rectangle", "cylinder"],
    "colours": ["green", "blue", "red", "yellow", "cyan", "magenta"],
    "textures": ["dotted", "plain", "stary"]
}
EVAL_CONFIG['task']['initializers']['workspace']['min_pose'] = [0.25, -0.5, 0.4]
EVAL_CONFIG['task']['initializers']['workspace']['max_pose'] = [0.6, 0.45, 0.4]
# PHYSICS CONFIG

EVAL_CONFIG["physics_dt"] = 0.001
EVAL_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["gains"]["position"]["kp"] = 300
EVAL_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["gains"]["orientation"]["kp"] = 800
EVAL_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["convergence"]["position_threshold"] = 5e-3
EVAL_CONFIG["robots"]["arm"]["controller_config"]["controller_params"]["convergence"]["orientation_threshold"] = 0.17

def generate_singular(noun_phrase: str) -> str:
    """generate singular noun"""
    if result:= inflect.engine().singular_noun(noun_phrase):
        return result
    return noun_phrase

def generate_plural(noun_phrase: str) -> str:
    """generate plural noun"""
    if result:= inflect.engine().plural_noun(noun_phrase):
        return result
    return noun_phrase

def generate_refexp(generator: Generator,
                    props_info: dict,
                    target_props: list[int], 
                    sample:bool=True,
                    pressupose_prob: float=0.5) -> tuple[str, list[int]]:
    """Generate a reference expression for the subset of a target props"""

    def get_same(labels: str) -> list[int]:
        props = []
        for prop, prop_info in props_info.items():
            if all(label in prop_info['labels'].data.values() for label in labels):
                props.append(prop)
        return props
    
    if random.random() > pressupose_prob:
        quant = "exists"
    else:
        quant = random.choice(["unique"])

    match quant:

        case "exists":
            prop = random.choice(target_props)
            des, _ = generator.generate_description(props_info[prop]['labels'], sample)
            refexp = inflect.engine().a(des)
            ref_props = [prop]
            return refexp, ref_props
        case "every":
            while True:
                prop = random.choice(target_props)
                des, labels = generator.generate_description(props_info[prop]['labels'], sample)
                ref_props = get_same(labels)
                if ref_props:
                    break
            refexp = f"every {des}"
            return refexp, ref_props
        case "unique":
            while True:
                prop = random.choice(target_props)
                des, labels = generator.generate_description(props_info[prop]['labels'], sample)
                ref_props = get_same(labels)
                if ref_props:
                    break
            article = f"the {inflect.engine().number_to_words(len(ref_props))}"
            if len(ref_props) > 1: # plural
                des = generate_plural(des)
            refexp = f"{article} {des}"
            return refexp, ref_props
        case _ :
            raise NotImplementedError(f"{quant} not implemented")

def filter_props(props_info:dict, targets:list[str], filter: Callable) -> list[int]:
    """filter our props by the filter function"""
    props = []
    for prop, prop_info in props_info.items():
        if prop in targets:
            continue

        if all(not filter(props_info[target]['bbox'],
                          prop_info['bbox']) for target in targets):
            props.append(prop)

    return props

def make_exists_refexp(refexp: str) -> str:
    """make exists (a/an) version of the provided refexp string"""
    p = inflect.engine()
    refexp_tokens = refexp.split()
    match refexp_tokens[0]:
        case "every":
            return p.a(generate_singular(" ".join(refexp_tokens[1:])))
        case "the":
            return p.a(generate_singular(" ".join(refexp_tokens[2:])))
        case "a" :
            return refexp
        case _:
            raise NotImplementedError(f"Cannot make exists version of {refexp}")
        
def make_every_refexp(refexp: str) -> str:
    """make every version of the provided refexp string"""
    refexp_tokens = refexp.split()
    match refexp_tokens[0]:
        case "every":
            return refexp
        case "the":
            return f"every {generate_singular(' '.join(refexp_tokens[2:]))}"
        case "a" :
            return f"every {generate_singular(' '.join(refexp_tokens[1:]))}"
        case _:
            raise NotImplementedError(f"Cannot make every version of {refexp}")
            
@dataclass
class Task:
    """Base task class for the agent"""

    def __init__(self, s_refexp:str, generator:Generator,) -> None:

        self.s_refexp = s_refexp
        self.s_refexp_lf = generator.parse_refexp(s_refexp)

        self.instruction = ""
        self.actions = [ActExecution()]

    def __str__(self):
        return f"Task: {self.instruction}"
    
    def __repr__(self) -> str:
        return f"Task: {self.instruction}"


class ResolutionTask(Task):

    def __init__(self, s_refexp:str, generator:Generator) -> None:
        super().__init__(s_refexp, generator)
 
        self.instruction = f"Show me {s_refexp}"
        self.actions = [ActExecution()]

    def __str__(self):
        return f"Task: {self.instruction}"
    
    def __repr__(self) -> str:
        return f"Task: {self.instruction}"

    @classmethod
    def from_props_info(cls, props_info: dict, generator: Generator, pressupose_prob: float=0.5):
        """Create a task from the properties information"""
        props = list(props_info.keys())
        refexp, _ = generate_refexp(generator, props_info, props, pressupose_prob)

        return cls(refexp, generator)
    
    @staticmethod
    def create_env_and_task(viewer: bool,
                            train: bool,
                            model_name: str,
                            generator: Generator,
                            pressupose_prob: float=0.5,
                            min_objects: int=4,
                            max_objects: int=9):

        CONFIG = TRAIN_CONFIG if train else EVAL_CONFIG
        # override the min and max objects
        CONFIG['arena']['props']['min_objects'] = min_objects
        CONFIG['arena']['props']['max_objects'] = max_objects
        # create environment
        try:
            env = RearrangementEnv(viewer,CONFIG)
            env.reset()
        except: # if there is an error in creating the due to e.g. camera problems
            env.close()
            del env
            gc.collect()
            return ResolutionTask.create_env_and_task(viewer, train, model_name, generator, pressupose_prob, min_objects, max_objects)
        
        model = make_domain_model(env.props_info)
        task = ResolutionTask.from_props_info(env.props_info, generator, pressupose_prob)
        records = extract_records(env, model_name=model_name)
        return env, task, records, model
    
    
    @staticmethod
    def create_env_and_n_task(viewer: bool,
                            train: bool,
                            model_name: str,
                            generator: Generator,
                            n_tasks: int,
                            pressupose_prob: float=0.5,
                            min_objects: int=4,
                            max_objects: int=9):
        """Create n tasks under the same environment"""

        CONFIG = TRAIN_CONFIG if train else EVAL_CONFIG
        # override the min and max objects
        CONFIG['arena']['props']['min_objects'] = min_objects
        CONFIG['arena']['props']['max_objects'] = max_objects
        # create environment
        try:
            env = RearrangementEnv(viewer,CONFIG)
            env.reset()
        except: # if there is an error in creating the due to e.g. camera problems
            env.close()
            del env
            gc.collect()
            return ResolutionTask.create_env_and_n_task(viewer, train, model_name, generator, pressupose_prob, min_objects, max_objects)
        
        model = make_domain_model(env.props_info)
        records = extract_records(env, model_name=model_name)
        # make sure all tasks have different s_refexp
        tasks = []
        while len(tasks) < n_tasks:
            task: ResolutionTask = ResolutionTask.from_props_info(env.props_info, generator, pressupose_prob)
            if task.s_refexp not in [t.s_refexp for t in tasks]:
                tasks.append(task)
    
        return env, tasks, records, model
    

class InteractiveResolutionTask(ResolutionTask):

    def __init__(self, s_refexp:str, generator: Generator) -> None:
        super().__init__(s_refexp, generator)

        self.s_refexp_exists = make_exists_refexp(s_refexp)
        self.s_refexp_exists_lf = generator.parse_refexp(self.s_refexp_exists)
        self.s_refexp_every = make_every_refexp(s_refexp)
        self.s_refexp_every_lf = generator.parse_refexp(self.s_refexp_every)
        

        self.actions = list(set([
            ActExecution(),
            Query(self.s_refexp, self.s_refexp_lf),
            Query(self.s_refexp_exists, self.s_refexp_exists_lf),
            Query(self.s_refexp_every, self.s_refexp_every_lf)
        ]))

    @classmethod
    def from_props_info(cls, props_info: dict, generator: Generator, pressupose_prob: float=0.5):
        """Create a task from the properties information"""
        props = list(props_info.keys())
        refexp, _ = generate_refexp(generator, props_info, props, pressupose_prob)

        return cls(refexp, generator)
    
    @staticmethod
    def create_env_and_task(viewer: bool,
                            train: bool,
                            model_name: str,
                            generator: Generator,
                            pressupose_prob: float=0.5,
                            min_objects: int=4,
                            max_objects: int=9):

        CONFIG = TRAIN_CONFIG if train else EVAL_CONFIG
        # override the min and max objects
        CONFIG['arena']['props']['min_objects'] = min_objects
        CONFIG['arena']['props']['max_objects'] = max_objects
        # create environment
        try:
            env = RearrangementEnv(viewer,CONFIG)
            env.reset()
        except: # if there is an error in creating the due to e.g. camera problems
            env.close()
            del env
            gc.collect()
            return InteractiveResolutionTask.create_env_and_task(viewer, train, model_name, generator, pressupose_prob, min_objects, max_objects)
        
        model = make_domain_model(env.props_info)
        task = InteractiveResolutionTask.from_props_info(env.props_info, generator,pressupose_prob)
        records = extract_records(env, model_name=model_name)
        return env, task, records, model
    
    @staticmethod
    def create_env_and_n_task(viewer: bool,
                            train: bool,
                            model_name: str,
                            generator: Generator,
                            n_tasks: int,
                            pressupose_prob: float=0.5,
                            min_objects: int=4,
                            max_objects: int=9):
        """Create n tasks under the same environment"""

        CONFIG = TRAIN_CONFIG if train else EVAL_CONFIG
        # override the min and max objects
        CONFIG['arena']['props']['min_objects'] = min_objects
        CONFIG['arena']['props']['max_objects'] = max_objects
        # create environment
        try:
            env = RearrangementEnv(viewer,CONFIG)
            env.reset()
        except: # if there is an error in creating the due to e.g. camera problems
            env.close()
            del env
            gc.collect()
            return InteractiveResolutionTask.create_env_and_n_task(viewer, train, model_name, generator, pressupose_prob, min_objects, max_objects)
        
        model = make_domain_model(env.props_info)
        records = extract_records(env, model_name=model_name)
        # make sure all tasks have different s_refexp
        tasks = []
        while len(tasks) < n_tasks:
            task: InteractiveResolutionTask = InteractiveResolutionTask.from_props_info(env.props_info, generator, pressupose_prob)
            if task.s_refexp not in [t.s_refexp for t in tasks]:
                tasks.append(task)
    
        return env, tasks, records, model
    
class RearrangementTask(InteractiveResolutionTask):
    """Utilities for specifying the (rearrangment) task for the agent"""

    def __init__(self,
                 s_refexp: str,
                 rel: str,
                 o_refexp: str,
                 generator: Generator) -> None:
        
        super().__init__(s_refexp, generator)

        self.rel = rel

        self.o_refexp = o_refexp
        self.o_refexp_exists = make_exists_refexp(o_refexp)
        self.o_refexp_every = make_every_refexp(o_refexp)

        self.o_refexp_lf = generator.parse_refexp(o_refexp)
        self.o_refexp_exists_lf = generator.parse_refexp(self.o_refexp_exists)
        self.o_refexp_every_lf = generator.parse_refexp(self.o_refexp_every)

        self.instruction =  f"Move {s_refexp} {rel} {o_refexp}"

        self.full_refexp = f"{s_refexp} {rel} {o_refexp}"
        self.full_refexp_lf = generator.parse_refexp(self.full_refexp)

        self.actions = list(set([
                ActExecution(),
                Query(self.s_refexp, self.s_refexp_lf),
                Query(self.s_refexp_exists, self.s_refexp_exists_lf),
                Query(self.s_refexp_every, self.s_refexp_every_lf),
                Query(self.o_refexp, self.o_refexp_lf),
                Query(self.o_refexp_exists, self.o_refexp_exists_lf),
                Query(self.o_refexp_every, self.o_refexp_every_lf),
            ]))
       
    @classmethod
    def from_props_info(cls,props_info: dict, generator: Generator):
        """Create a task from the properties information"""
        while True:
            props = list(props_info.keys())
            s_refexp, s_prop = generate_refexp(generator, props_info, props)

            # generate a random spatial relationship
            rel, rel_func = Spatial.sample()

            props = filter_props(props_info, s_prop, rel_func)
            
            if len(props) > 0:
                o_refexp, _ = generate_refexp(generator, props_info, props)
                # check if there are no dublicate works
                try:
                    return cls(s_refexp, rel, o_refexp, generator)
                except:
                    continue

    @staticmethod
    def create_env_and_task(viewer: bool,
                            train: bool,
                            model_name: str,
                            generator: Generator):

        CONFIG = TRAIN_CONFIG if train else EVAL_CONFIG
        # create environment
        try:
            env = RearrangementEnv(viewer,CONFIG)
            env.reset()
        except: # if there is an error in creating the due to e.g. camera problems
            env.close()
            del env
            gc.collect()
            return RearrangementTask.create_env_and_task(viewer, train, model_name, generator)
        
        model = make_domain_model(env.props_info)
        ## make sure to generate the task with disjoint reference objects
        task = RearrangementTask.from_props_info(env.props_info, generator)
        s_ent = Semantics.eval_refexp(task.s_refexp_lf, model).entities
        o_ent = Semantics.eval_refexp(task.o_refexp_lf, model).entities
        if len(set(s_ent) & set(o_ent)) > 0 and len(set(s_ent)) > 0 and len(set(o_ent)) > 0:
            # clean up the environment and call again
            env.close()
            del env
            gc.collect()
            return RearrangementTask.create_env_and_task(viewer, train, model_name, generator)
        else:
            records = extract_records(env, model_name=model_name)
            return env, task, records, model
