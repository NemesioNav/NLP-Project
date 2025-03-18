import ast
import os
import sys

import datasets
import numpy as np
import scipy.stats as sh
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from bert.BERT_utils import tokenize_ingredients

PATH = os.path.dirname(os.path.abspath(__file__))

if PATH not in sys.path:
    sys.path.append(PATH)

from bert.utils import evaluate_predictions


def valid_estimate(records):
    assert isinstance(records, list)
    assert len(records) > 0
    mu = np.mean(records)
    interval = sh.t.interval(
        confidence=0.9, df=len(records) - 1, loc=mu, scale=sh.sem(records)
    )
    criterion = (interval[1] - interval[0]) < 0.1
    return criterion


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.f1 = {
            "FOOD": [],
            "QUANTITY": [],
            "UNIT": [],
            "PROCESS": [],
            "PHYSICAL_QUALITY": [],
            "COLOR": [],
            "TASTE": [],
            "PURPOSE": [],
            "PART": [],
            "TRADE_NAME": [],
            "DIET": [],
            "EXAMPLE": [],
            "all": [],
        }

    def record_entities(self, references, predictions):
        assert len(references) == len(predictions)

        def is_list_of_list_of_str(x):
            output = isinstance(x, list)
            for y in x:
                output &= isinstance(y, list)
                for z in y:
                    output &= isinstance(z, str)
            return output

        assert is_list_of_list_of_str(references), references
        assert is_list_of_list_of_str(predictions), predictions
        for reference, prediction in zip(references, predictions):
            results = evaluate_predictions(
                [reference], [prediction], entities_format="bio"
            )
            self.record_metrics(results)

    def record_metrics(self, results):
        for entity_type in results:
            self.f1[entity_type].append(results[entity_type]["f1"])

    def get_metrics(self):
        def mean(l):
            return sum(l) / len(l)

        def get_confidence_interval(l):
            assert isinstance(l, list)
            assert len(l) > 0
            mu = mean(l)
            if sh.sem(l) == 0:
                return (mu, mu)
            else:
                interval = sh.t.interval(
                    confidence=0.9, df=len(l) - 1, loc=mu, scale=sh.sem(l)
                )
                interval = max(0, interval[0]), min(1, interval[1])
            return (interval[0], interval[1])

        output = {"f1": {}}
        for entity_type in self.f1:
            output["f1"][entity_type] = {
                "confidence_interval": get_confidence_interval(self.f1[entity_type])
            }
        output["N"] = len(self.f1["all"])
        return output


class FewShotRetriever:
    def __init__(self, data_set):
        # data_set contains items from the training set
        assert isinstance(data_set, datasets.arrow_dataset.Dataset)
        assert all(isinstance(x, dict) for x in data_set)
        self.data = [(x["ingredients"], x["entities"]) for x in data_set]
        self.bm25 = None

    def build_index(self):
        self.corpus = [x[0] for x in self.data]
        self.bm25 = BM25Okapi(self.corpus)

    def get_samples(self, query, n=3):
        top_n = self.bm25.get_top_n(query, self.data, n=n)
        return top_n


class LLMEntityExtractor:
    def __init__(self, model_name, retriever, n_few_shot=2, max_tokens=900):
        assert isinstance(retriever, FewShotRetriever)
        self.retriever = retriever
        assert isinstance(max_tokens, int)
        self.max_tokens = max_tokens
        assert isinstance(model_name, str)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, pad_token_id=self.tokenizer.eos_token_id
        )
        assert isinstance(n_few_shot, int)
        self.n_few_shot = n_few_shot

    def generate(self, input_strings):
        assert isinstance(input_strings, list)
        assert len(input_strings) > 0
        assert all(isinstance(ii, str) for ii in input_strings)
        prompts = self.make_prompts(input_strings)
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        generation_args = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": False,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "stop_strings": ["\n"],
            "tokenizer": self.tokenizer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        output = pipe(prompts, **generation_args)
        output = [oo[0] for oo in output]
        assert len(output) == len(prompts)
        output = [
            {"prompt": prompt, "lm_output": x["generated_text"]}
            for x, prompt in zip(output, prompts)
        ]
        return output

    def make_prompt(self, input_string):
        samples = self.retriever.get_samples(input_string, n=self.n_few_shot)
        # assert len(samples) == self.n_few_shot
        prompt = []
        for x, y in samples:
            x = tokenize_ingredients(x)
            y = list(zip(x, y))
            prompt.append("{}\n{}".format(x, y))
        prompt.append("{}\n".format(tokenize_ingredients(input_string)))
        prompt = "\n".join(prompt)
        return prompt

    def make_prompts(self, input_strings):
        prompts = [self.make_prompt(input_string) for input_string in input_strings]
        return prompts

    def eval(self, data, batch_size=10, variable_size=False):
        if variable_size:
            return self.eval_variable_size(data, batch_size=batch_size)
        metrics = Metrics()
        metrics.reset()
        for position in tqdm(range(0, len(data), 10)):
            chunk = data[position : position + 10]
            assert len(chunk) > 0
            self.eval_step(chunk, metrics)
        return metrics.get_metrics()

    def eval_variable_size(self, data, batch_size=10):
        metrics = Metrics()
        metrics.reset()
        position = 0
        terminate = False
        while (position < len(data)) and (not terminate):
            chunk = data[position : position + batch_size]
            assert len(chunk) > 0
            self.eval_step(chunk, metrics)
            position += batch_size
            terminate = valid_estimate(metrics.f1["all"])
        return metrics.get_metrics()

    def eval_step(self, chunk, metrics):
        y = self.generate(chunk["ingredients"])

        outputs_expected = []
        outputs_extracted = []
        for i, out in enumerate(y):
            try:
                out_parsed = ast.literal_eval(out["lm_output"].strip())
                generated_entities = [t[1] for t in out_parsed]
            except:
                continue
            outputs_extracted.append(generated_entities)
            outputs_expected.append(chunk["entities"][i])

        metrics.record_entities(outputs_expected, outputs_extracted)
