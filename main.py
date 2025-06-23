import sys
import gc
import random
import time

import wandb
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from config import SECUREConfig
from logic_toolkit import (Symbol, Reasoning, Semantics,
                           DomainModel)
import torch
from torch import Tensor
import numpy as np
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

from secure.generator import Generator
from secure.task import create_env_and_task, Task
from secure.utils import make_init_theory, Spatial, task_f1_score, process_correction
from secure.learner import Learner
from secure.messages import (Complete, ActExecution, Message, 
                             PickMove, PlaceMove, Correction, 
                             Silence, CompleteResponce, Query)
from secure.oracle import Oracle
from secure.belief import Belief
from secure.grounder import Grounder
from secure.reward import reward


def logging(log:bool,
            task_idx: int,
            task: Task, 
            step: int,
            qs: dict[Message, Tensor],
            gains: dict[Message, Tensor],
            exp_rewards: dict[Message, Tensor],
            action: Message,
            responce: Message,
            env: RearrangementEnv,
            init_model: DomainModel,
            pred_model: DomainModel,
            reward: Tensor) -> None:
    """Logging the task progress"""

    if log:
    
        f1_score = task_f1_score(pred_model,
                                init_model,
                                [task.s_refexp_lf, task.o_refexp_lf])
        
        obs = env._compute_observation()['overhead_camera/rgb']

        # use qs, gains, exp_rewards to make wandb table
        q_table = wandb.Table(columns=["action", "q value", "info gain", "exp reward"])
        for action, q_value in qs.items():
            q_table.add_data(str(action),
                            float(q_value),
                            float(gains[action]),
                            float(exp_rewards[action]))

        wandb.log({
            "task_idx": task_idx,
            "task": wandb.Html(str(task)),
            "step": step,
            "q_values": q_table,
            "action": wandb.Html(str(action)),
            "responce": wandb.Html(str(responce)),
            "observation": wandb.Image(obs),
            "f1_score": f1_score*100,
            "reward": reward,
        })

def main(cfg: SECUREConfig) -> None:

    # Initialize wandb
    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        # print the config
        print(OmegaConf.to_yaml(cfg))

    # seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    generator = Generator(
        model_name=cfg.generator.model_name,
        client=cfg.generator.client,
        temperature=cfg.generator.temperature,
        top_k=cfg.generator.top_k,
        do_sample=cfg.generator.do_sample,
        num_beams=cfg.generator.num_beams,
        max_new_tokens=cfg.generator.max_new_tokens,
    )

    grounder = Grounder(
        memory={},
        vocab=[Symbol("object",1)],
        threshold=cfg.grounder.threshold,
        feat_size=cfg.grounder.feat_size,
    )
    
    learner = Learner(
        name=cfg.learner.name,
        learning_rate=cfg.learner.learning_rate,
        epsilon=cfg.learner.epsilon,
        gamma=cfg.learner.gamma,
        init_weights=cfg.learner.init_weights,
        point_cost=cfg.learner.point_cost,
        symbol_cost=cfg.learner.symbol_cost,
        update_freq=cfg.learner.update_freq,
        semaware=cfg.learner.semaware,
        dialogue=cfg.learner.dialogue,
        load_path=cfg.learner.load_path,
        save_path=cfg.learner.save_path,
        save=cfg.learner.save,
        load=cfg.learner.load,
    )

    task_idx = 0

    while task_idx < cfg.env.num_tasks:

        try:

            # initialize the environment and task
            env, task, records, init_model = create_env_and_task(
                cfg.env.viewer,
                cfg.train,
                cfg.grounder.model_name,
                generator,
            )

            entities = init_model.entities

            grounder.add_records(task_idx, records)
            # constuct belief state and domain theory
            exps, task_vocab = make_init_theory(init_model,
                                                task.s_refexp_lf,
                                                task.o_refexp_lf)
                    
            belief = Belief(
                idx=task_idx,
                exps=exps,
                records=records,
                vocab=task_vocab,
                grounder=grounder,
                priors=cfg.belief.priors,
                base=cfg.belief.base,
                semaware=cfg.belief.semaware,
            )

            oracle = Oracle(
                env=env,
                task=task,
                explain=cfg.oracle.explain,
                name=cfg.oracle.name,
                generator=generator,
            )

            complete_move = Complete()

            responce = oracle.responce(complete_move)
            
            learner.set_task(task)
            step = 0
            task_attempts = 0
            
            while task_attempts < cfg.env.max_attempts:

                pred_model = belief.map
                qs, gains, exp_rewards = learner.q_values(belief, learner.params)
                action = learner.choose_action(qs)
                step += 1
                
                # choosing to perform pick-and-place moves with predicted model
                if isinstance(action, ActExecution):

                    #attempt to solve the task
                    task_attempts += 1

                    responce = oracle.responce(action)

                    logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, action, responce, env, init_model, pred_model, reward=0.0)

                    # compute subject and object props
                    s_props = [int(s.name) for s in Semantics.eval_refexp(task.s_refexp_lf, pred_model).entities]
                    o_props = [int(o.name) for o in Semantics.eval_refexp(task.o_refexp_lf, pred_model).entities]

                    # workspace region bounded by the spatial relationship
                    r_max_pose, r_min_pose = Spatial.get_region_poses(o_props, task.rel, env, cfg.env.camera)
                    # execute the pick-and-place moves
                    for prop in s_props:
                    
                        pick_spec = env.prop_pick(prop)
                        place_spec = env.prop_place(prop, r_min_pose, r_max_pose)

                        # PICK MOVE
                        step += 1
                        env.pick(pick_spec)
                        pick_move = PickMove(prop)
                        
                        match responce:= oracle.responce(pick_move):

                            case Correction():
                                print(f"correction for pick move: {pick_move} with responce: {responce}")
                                # Correction. Return to original pose
                                env.place(pick_spec)
                                # build exp and update belief
                                exp = process_correction(d_refexp=responce.refexp_lf, # designation refexp
                                                        t_refexp=task.s_refexp_lf,   # truth-condition refexp
                                                        ref=responce.referent,
                                                        entities=entities)
                                belief.update(exp)
                                r = reward(action, responce, cfg.learner.symbol_cost, cfg.learner.point_cost, entities)
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                                break
                            case Silence():
                                r = 0
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                            case _:
                                raise ValueError(f"not supported responce: {responce}")
                        


                        # PLACE MOVE
                        step += 1
                        env.place(place_spec)
                        place_move = PlaceMove(prop)

                        match responce:= oracle.responce(place_move):

                            case Correction():
                                print(f"correction for place move: {place_move} with responce: {responce}")
                                # Correction. Return to original pose
                                env.pick(place_spec)
                                env.place(pick_spec)
                                # build exp and update belief
                                exp = process_correction(d_refexp=responce.refexp_lf, # designation refexp
                                                        t_refexp=task.o_refexp_lf, # truth-condition refexp
                                                        ref=responce.referent,
                                                        entities=entities)
                                belief.update(exp)
                                r = reward(action, responce, cfg.learner.symbol_cost, cfg.learner.point_cost, entities)
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                                break
                            case Silence():
                                r = 0
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                            case _:
                                raise ValueError(f"not supported responce: {responce}")

                    else:
                        """if all subject props are placed, then declare completion and check with oracle if correct."""
                        step += 1
                        complete_move = Complete()

                        match responce:= oracle.responce(complete_move):
                            case CompleteResponce():
                                r = reward(complete_move, responce, cfg.learner.symbol_cost, cfg.learner.point_cost, entities)
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                            case Correction():
                                # build exp and update belief
                                exp = Reasoning.process_refexp(responce.refexp_lf, responce.referent, entities)
                                belief.update(exp)
                                # reward for correction
                                r = reward(complete_move, responce, cfg.learner.symbol_cost, cfg.learner.point_cost, entities)
                                logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, pick_move, responce, env, init_model, pred_model, r)
                            case _:
                                raise ValueError(f"not supported responce: {responce}")
                            
                elif isinstance(action, Query):
                    # query oracle
                    responce = oracle.responce(action)
                    # build exp and update belief
                    exp = Reasoning.process_refexp(action.refexp_lf, responce.referent, entities, not cfg.belief.semaware)
                    belief.update(exp)
                    # reward for query
                    r = reward(action, responce, cfg.learner.symbol_cost, cfg.learner.point_cost, entities)
                    
                    logging(cfg.wandb.use_wandb, task_idx, task, step, qs, gains, exp_rewards, action, responce, env, init_model, pred_model, r)
                else:
                    raise ValueError(f"not supported action: {action}")

                if cfg.train:
                    learner.train(qs[action], action, r, belief)

                    if cfg.learner.save:
                        learner.save_params(f"{cfg.learner.name}-{cfg.seed}-{task_idx}")

                # check if the task is completed: close the environment and move to the next task
                if isinstance(responce, CompleteResponce):
                    task_idx += 1
                    env.close()
                    del env, oracle
                    gc.collect()
                    break
        
        except Exception as e:
            print(f"Error: {e}")
            env.close()
            del env, oracle
            gc.collect()
            continue


    # Close wandb
    if cfg.wandb.use_wandb:
        wandb.finish()
        

if __name__ == '__main__':

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="config", job_name="secure_config")
    cfg = compose(config_name="secure", overrides=sys.argv[1:])

    main(cfg)