from dataclasses import dataclass
from logic_toolkit import Symbol


@dataclass
class WandbConfig:
    use_wandb: bool
    project: str 
    entity: str 
    name: str


@dataclass
class GeneratorConfig:
    model_name: str
    client: str|None
    temperature: float
    top_k: int
    do_sample: bool
    num_beams: int
    max_new_tokens: int


@dataclass
class PriorWeightConfig:

    def __init__(self, name: str, weight: float) -> None:
        self.symbol: Symbol = Symbol(name, 1)
        self.weight: float = weight


@dataclass
class BeliefConfig:
    base: float
    priors: list[PriorWeightConfig]
    semaware: bool


@dataclass
class GrounderConfig:
    model_name: str
    threshold: float
    feat_size: int

@dataclass
class LearnerConfig:
    name: str
    point_cost: float
    symbol_cost: float
    semaware: bool
    dialogue: bool
    load_path: str
    save_path: str
    save: bool
    load: bool
    learning_rate: float
    gamma: float
    epsilon: float
    init_weights: list[float]
    update_freq: int


@dataclass
class OracleConfig:
    name: str
    explain: bool


@dataclass
class EnvironmentConfig:
    viewer: bool
    num_tasks: int
    max_attempts: int
    camera: str
    num_tasks_per_env: int
    presupose_prob: float
    min_objects: float
    max_objects: float


@dataclass
class SECUREConfig:
    seed: int
    train: bool
    wandb: WandbConfig
    generator: GeneratorConfig
    belief: BeliefConfig
    grounder: GrounderConfig
    learner: LearnerConfig
    oracle: OracleConfig
    env: EnvironmentConfig