import re
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import InferenceClient
from logic_toolkit import RefExp, RefExpParser
from mujoco_robot_environments.environment.props import PropsLabels

PARSE_REFEXP_TEMPLATE = """[INST] You are a helper in a virtual environment. You are asked to translate referential expressions to their logical forms.
Referential Expressions like "the one red block" has a logical form "<_the_1_q x. red(x) ^ block(x)>".
Logical forms follows the pattern "<[Quantifier] [Variable]. [LogicFormula]>"

[Quantifier] is a quantifier. We consider the following quantifiers:
- existential with surface form: "a/an" and denoted as logic symbol: "_a_q"
- universal with surface form: "every/all" and denoted as logic logic symbol: "_every_q"
- uniqness with surface form: "the n" and denoted as logic symbol: "_the_n_q", where n is a natural number like "one", "two", etc.

[Variable] is a variable e.g. x, x1, x12. Note that [Variable] is the only free variable in [LogicFormula] not bound by a quantifier. 
if formula has only one variable, it should be named x.

[LogicFormula] is formula of predicate logic. Each formula is constructed recusively:
- Predicates like "red(x)", "above(x1,x2)", "left(x1,x3)" is a well-formed formula [LogicFormula]
- Negation of [LogicFormula] like "neg(red(x1))" is a well-formed formula [LogicFormula]
- Logical conjuction of [LogicFormula]s like "red(x) ^ block(x)" is a well-formed formula [LogicFormula]
- Logical disjunction of [LogicFormula]s like "red(x) v block(x)" is a well-formed formula [LogicFormula]
- Logical implication of [LogicFormula]s like "red(x) -> block(x)" is a well-formed formula [LogicFormula]
- Structure like [Quantifier][Variable].([LogicFormula],[LogicFormula]) like "_the_1_q x. (red(x), block(x))" is a well-formed formula [LogicFormula]

Here is some examples of how these logical forms of referential expressions looks like:

|RefExp|LF-RefExp|
"a block.", "<_a_q x. block(x)>"
"the one block.", "<_the_1_q x. block(x)>"
"the two plain objects.", "<_the_2_q x. plain(x) ^ object(x)>"
"every magenta sphere." "<_every_q x. magenta(x) ^ sphere(x)>"
"not a block above a sphere.", "<_a_q x.neg( _a_q x1.(sphere(x1),block(x) ^ above(x,x1)))>"
"a sphere to the left of every green cone.", "<_a_q x. _every_q x1.(green(x1) ^ cone(x1),sphere(x) ^ left(x,x1))>",
"every sphere to the left of every green object.", "<_every_q x. _every_q x1.(green(x1) ^ object(x1),sphere(x) ^ left(x,x1))>",
"a sphere to the right of the two green cones.", "<_a_q x. _the_2_q x1.(green(x1) ^ cone(x1),sphere(x) ^ right(x,x1))>",
"the one sphere in front of every green cone.", "<_the_1_q x. _every_q x1.(green(x1) ^ cone(x1),sphere(x) ^ front(x,x1))>",
"the two spheres behind a green cone.", "<_the_2_q x. _a_q x1.(green(x1) ^ cone(x1),sphere(x) ^ behind(x,x1))>",

Now, please translate the following referential expression to its logical form. Just give the logical form. No extra information.
Referential expression: {refexp} [/INST]"""


GEN_NP_TEMPLATE = """[INST] Given a list of words to for a English description return them in the correct order as they would appear in an English description. 
The correct order should follow the typical adjective order: Quantity, Quality, Size, Age, Shape, Color, Proper adjective (often nationality, other place of origin, or material), Purpose or qualifier.
You return a single string and nothing else. For example: 
[red, sphere, dotted] -> "dotted red sphere"
[blue, cone, plain] -> "plain blue cone"
[cone, red] -> "red cone"
[green, sphere, small] -> "small green sphere"
[stary, purple, block] -> "stary purple block"
[plain, green, cube] -> "plain green cube"
[plain, cube, green] -> "plain green cube"
[cube, plain, green] -> "plain green cube"
[red, sphere, big] -> "big red sphere"
[blue, rectangle, dotted] -> "dotted blue rectangle"
[green, sphere, small] -> "small green sphere"
Now do yourself: 
[{words}] -> [/INST]"""



class Generator:

    def __init__(self,
                 model_name:str,
                 client: str,
                 temperature: float,
                 top_k: int,
                 do_sample: bool,
                 num_beams: int,
                 max_new_tokens: int) -> None:
        """LLM-based generator for varioust tasks"""

        self.model_name = model_name
        self.client = client
        self.temperature = temperature
        self.top_k = top_k
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        if client is None:
            """Initialize the model and tokenizer locally."""
            self.client = None 

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        device_map="auto",
                                                        quantization_config=bnb_config,
                                                        torch_dtype=torch.float16,
                                                        trust_remote_code=False,
                                                        revision="main")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    use_fast=True)
        else:
            self.client = InferenceClient(model=client)


    def parse_refexp(self, refexp:str) -> RefExp:

        def preprocess(text:str) -> str:
            """prepare input for generation"""
            if text.endswith("."):
                return text
            else:
                return text + "."

        def postprocess(text:str) -> RefExp:
            """Clean up generation results and return the logical form."""
            refexp_lf_str = re.search(r'<[^>]+>',text).group()
            try:
                return RefExpParser()(refexp_lf_str)
            except:
                raise ValueError(f"Failed to parse logical form: {refexp_lf_str}")
        
        prompt = PARSE_REFEXP_TEMPLATE.format(refexp=preprocess(refexp))

        if self.client is None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                        attention_mask=inputs["attention_mask"].to("cuda"),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        temperature=self.temperature,
                                        top_k=self.top_k,
                                        do_sample=self.do_sample,
                                        num_beams=self.num_beams,
                                        max_new_tokens=self.max_new_tokens)
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            output = self.client.text_generation(prompt=prompt,
                                                temperature=self.temperature,
                                                do_sample=self.do_sample,
                                                top_k=self.top_k,
                                                best_of = self.num_beams,
                                                max_new_tokens=self.max_new_tokens)
        return postprocess(output)



    def generate_description(self, labels: PropsLabels, sample:bool=True) -> tuple[str,list[str]]:
        """generate noun phrase (description) from the list of words"""

        def preprocess(labels: PropsLabels, sample:bool=True) -> str:

            description = list(set(labels.data.values()))
            
            if sample:
                # choose a random subset of labels
                description = random.sample(description, 
                                            random.randint(1,len(description)))
                # if labels does not have a shape, add "object" to the list
                if labels.shape not in description:
                    description.append("object")

            return f"[{', '.join(description)}]", description


        def postprocess(text:str) -> str:
            """Clean up generation results and return the logical form."""
            match = re.search(r'"(.*?)"', text)
            if match:
                return match.group(1)
            else:
                return text

        words, labels = preprocess(labels, sample)
        prompt = GEN_NP_TEMPLATE.format(words=words)

        output = self.client.text_generation(prompt=prompt,
                                        temperature=self.temperature,
                                        do_sample=self.do_sample,
                                        top_k=self.top_k,
                                        best_of =self.num_beams,
                                        max_new_tokens=self.max_new_tokens)
        
        return postprocess(output), labels
