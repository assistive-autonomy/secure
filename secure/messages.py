from dataclasses import dataclass

from logic_toolkit import RefExp, Referent, Entity, Denotation, DenotationSet


@dataclass(eq=True)
class Message:
    """Dialogue message container"""
    move: str
    agent: str
    text: str = ""

    def __str__(self) -> str:
        return f"[{self.move}] {self.agent}: {self.text}"

    def __repr__(self) -> str:
        return f"[{self.move}] {self.agent}: {self.text}"

    def __hash__(self) -> int:
        return hash(str(self))    


class NotSupportedMessageError(Exception):
    def __init__(self, msg: Message) -> None:
        super(NotSupportedMessageError, self).__init__(f"not supported message: {msg}")


class Instruction(Message):

    def __init__(self,
                 instruction: str,
                 agent: str = "teacher") -> None:
        super().__init__("INSTRUCTION", agent, instruction)
        self.instruction = instruction


class Query(Message):

    def __init__(self,
                 refexp_str: str,
                 refexp_lf: RefExp,
                 agent:str = "learner") -> None:
        super().__init__("QUERY", agent, f"Before this, show me {refexp_str}")
        self.refexp_lf = refexp_lf


class QueryResponse(Message):

    def __init__(self,
                 referent: Referent,
                 agent:str = "teacher") -> None:
        super().__init__("QUERY", agent, f"Here it is. (point to {referent})")
        self.referent = referent


class Clarification(Message):

    def __init__(self,
                 refexp_str: str,
                 refexp_lf: RefExp,
                 referent: Referent = None,
                 agent:str = "learner") -> None:
        super().__init__("CLARIFICATION", agent, f"Before this, is this {refexp_str} (points to {referent})")
        self.refexp_str = refexp_str
        self.refexp_lf = refexp_lf
        self.referent = referent


class ClarificationResponse(Message):

    def __init__(self, 
                 correct: bool,
                 referent: Referent,
                 agent:str = "teacher") -> None:
        self.correct = correct
        self.referent = referent
        if correct:
            super().__init__("CLARIFICATION", agent, "Yes.")
        else:
            super().__init__("CLARIFICATION", agent, "No.")


class ActExecution(Message):

    def __init__(self,
                 agent:str = "learner") -> None:
        super().__init__("ACT", agent, f"Okay. Lets try to achieve the task.")


class Silence(Message):

    def __init__(self,
                 agent:str = "teacher") -> None:
        super().__init__("SILENCE", agent, "...")


class PickMove(Message):

    def __init__(self,
                 prop_id: int,
                 agent:str = "learner") -> None:
        super().__init__("PICK", agent, f"(picks {prop_id})")
        self.prop_id = prop_id
        self.entity = Entity(str(prop_id))


class PlaceMove(Message):

    def __init__(self,
                 prop_id: int,
                 agent:str = "learner") -> None:
        super().__init__("PLACE", agent, f"(place {prop_id})")
        self.prop_id = prop_id
        self.entity = Entity(str(prop_id))


class Complete(Message):

    def __init__(self,
                 agent:str = "learner") -> None:
        super().__init__("COMPLETE", agent, "I have finished the task.")


class CompleteResponse(Message):

    def __init__(self,
                 agent:str = "teacher") -> None:
        super().__init__("COMPLETE", agent, "You have executed the task correctly.")


class Resolution(Message):

    def __init__(self,
                 referent: Referent,
                 agent:str = "learner") -> None:
        super().__init__("COMPLETE", agent, f"I think you are looking for this (points to {referent})")
        self.referent = referent

class ResolutionResponse(Message):

    def __init__(self,
                 correct: bool,
                 referent: Referent,
                 agent:str = "learner") -> None:
        if correct:
            super().__init__("COMPLETE", agent, "Correct.")
        else:
            super().__init__("COMPLETE", agent, f"No. I wanted this (points to {referent})")

        self.correct = correct
        self.referent = referent


class Correction(Message):

    def __init__(self,
                 prop_id: int = None,
                 refexp: str = None,
                 refexp_lf: RefExp = None,
                 agent:str = "teacher") -> None:
        if refexp is not None and prop_id is not None:
            correction = f"No. This is {refexp} (points to {prop_id})"
        else:
            correction = "No."
        super().__init__("CORRECTION", agent, correction)
        self.prop_id = prop_id
        self.referent = Referent(DenotationSet(Denotation(Entity(str(prop_id)))))
        self.refexp = refexp
        self.refexp_lf = refexp_lf
