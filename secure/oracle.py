import random

import inflect
from logic_toolkit import DomainModel, Semantics, Entity, Referent
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv


from secure.messages import (
    Instruction, ActExecution, Query, QueryResponse, Clarification, ClarificationResponse,
    Silence, PickMove, PlaceMove, Correction, Resolution, ResolutionResponse,
    Complete, CompleteResponse, Message, NotSupportedMessageError,
)
from secure.task import Task, ResolutionTask, InteractiveResolutionTask, RearrangementTask
from secure.generator import Generator
from secure.utils import Spatial, make_domain_model

class Oracle:
    """"Oracle/teacher providing the feedback to the agent in interaction"""

    def __init__(self,
                env: RearrangementEnv,
                task: Task,
                generator: Generator,
                explain: bool = False,
                name: str = "teacher"):
    
        self.env = env
        # asssociate prop_id with labels. This is not fluent so will not change during in simulation
        self.labels = {prop_id: self.env.props_info[prop_id]['labels'] for prop_id in self.env.props_info}
        self.task = task
        self.s_ref = Semantics.eval_refexp(task.s_refexp_lf, self.model)
        if isinstance(task, RearrangementTask):
            self.o_ref = Semantics.eval_refexp(task.o_refexp_lf, self.model)
        else:
            self.o_ref = None
        self.generator = generator
        self.explain = explain
        self.name = name

    @property
    def props_info(self) -> dict:
        return self.env.props_info
    
    @property
    def model(self) -> DomainModel:
        return make_domain_model(self.props_info)

    def update_task(self, task: RearrangementTask) -> None:
        self.task = task

    def start_conversation(self) -> Instruction:
        return Instruction(self.task, self.name)
    
    def query_response(self, msg: Query) -> QueryResponse:

        referent = Semantics.eval_refexp(msg.refexp_lf, self.model)
        # sample one Denotation set from the referent
        referent = Referent(random.choice(list(referent._data)))

        return QueryResponse(referent, self.name)
    
    def clarification_response(self, msg: Clarification) -> ClarificationResponse:

        referent = Semantics.eval_refexp(msg.refexp_lf, self.model)
        correct = referent == msg.referent

        return ClarificationResponse(correct, referent, self.name)

    def act_response(self, msg: ActExecution) -> Silence:
        """response to act is silence as no move action observed"""
        return Silence(self.name)
    
    def pick_response(self, msg: PickMove) -> Silence | Correction:
        """Pick move gets a response if the entity picked is not in refexp1"""
        if Entity(str(msg.prop_id))  in self.s_ref.entities:
            return Silence(self.name)
        if self.explain:
            labels = self.labels[msg.prop_id]
            description, _ = self.generator.generate_description(labels, sample=False)
            refexp = inflect.engine().a(description)
            refexp_lf = self.generator.parse_refexp(refexp)
            return Correction(msg.prop_id, refexp, refexp_lf, self.name)
        else:
            return Correction(self.name)

    def place_response(self, msg: PlaceMove) -> Silence | Correction:
        # get entities denoted by refexp2
        o_entities = self.o_ref.entities
        # gett bbox current entities
        s_bbox = self.props_info[msg.prop_id]['bbox']
        o_bboxs = [self.props_info[int(entity.name)]['bbox'] for entity in o_entities]
        
        if any(Spatial.eval(self.task.rel,s_bbox, o_bbox) for o_bbox in o_bboxs):
            return Silence(self.name)
        else:
            if self.explain:
                prop_id = int(random.choice(list(o_entities)).name)
                labels = self.labels[msg.prop_id]
                description, _ = self.generator.generate_description(labels, sample=False)
                refexp = inflect.engine().a(description)
                refexp_lf = self.generator.parse_refexp(refexp)
                return Correction(prop_id,refexp,refexp_lf,self.name)
            else:
                return Correction(self.name)

    def complete_response(self, msg: Complete) -> CompleteResponse | Correction:

        s_props = [int(entity.name) for entity in self.s_ref.entities]
        # o_props = [int(entity.name) for entity in self.o_ref.entities]

        # s_bboxs = [self.props_info[int(entity.name)]['bbox'] for entity in self.s_ref.entities]
        o_bboxs = [self.props_info[int(entity.name)]['bbox'] for entity in self.o_ref.entities]

        for s_prop in s_props:

            s_bbox = self.props_info[s_prop]['bbox']

            if not any(Spatial.eval(self.task.rel, s_bbox, o_bbox) for o_bbox in o_bboxs):
                if self.explain:
                    labels = self.labels[s_prop]
                    description, _ = self.generator.generate_description(labels, sample=False)
                    refexp = inflect.engine().a(description)
                    refexp_lf = self.generator.parse_refexp(refexp)
                    return Correction(s_prop, refexp, refexp_lf, self.name)
                else:
                    return Correction(self.name)
        
        return CompleteResponse(self.name)
    
    def resolution_response(self, msg: Resolution) -> ResolutionResponse:
        """response to the resolution message"""

        correct = self.s_ref == msg.referent
        return ResolutionResponse(correct, self.s_ref, self.name)
    
    def response(self, msg: Message) -> Message:
        """response to the message from the agent."""

        match msg:

            case Resolution():
                return self.resolution_response(msg)
            case Query():
                return self.query_response(msg)
            case Clarification():
                return self.clarification_response(msg)
            case ActExecution():
                return self.act_response(msg)
            case PickMove():
                return self.pick_response(msg)
            case PlaceMove():
                return self.place_response(msg)
            case Complete():
                return self.complete_response(msg)
            case _:
                raise NotSupportedMessageError(msg)
            