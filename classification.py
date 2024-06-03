import json
from abc import ABC, abstractmethod

import torch
import transformers
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,)

from utils import find_first_digit


access_config_path = 'config/access_config.json'
llama3_config_path = 'config/models/llama3_config.json'
llama2_config_path = 'config/models/llama2_config.json'
mistral_config_path = 'config/models/mistral_config.json'
mpt_config_path = 'config/models/mpt_config.json'
vicuna_config_path = 'config/models/vicuna_config.json'


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def label_to_id(label, id_dict, failure_id=-1):
    if label is not None:
        if label in id_dict.keys():
            return id_dict[label]
    return failure_id


def remove_prompt_from_output(output, prompt_len):
    gen_text = output[0]["generated_text"][prompt_len:]
    return gen_text


class NarrativeClassifier(ABC):
    @abstractmethod
    def classify(self, inputs, narrative_dict):
        pass


class CasualLMClassifier(NarrativeClassifier):

    def __init__(self, pipeline, label_extraction_function=find_first_digit, temperature=0.0, do_sample=True):
        self.pipeline = pipeline
        self.extract_label = label_extraction_function
        self.temperature = temperature
        self.do_sample = do_sample

    def classify(self, inputs, narrative_id_dict):
        outputs = self.pipeline(inputs, temperature=self.temperature, do_sample=self.do_sample)

        prompt_lengths = list(map(len, inputs))
        responses = list(map(remove_prompt_from_output, outputs, prompt_lengths))
        prompt_labels = map(self.extract_label, responses)
        narrative_id_labels = list(map(lambda x: label_to_id(x, narrative_id_dict), prompt_labels))
        return narrative_id_labels, responses


def build_casual_lm_classifier(model_config, access_config, label_extraction_function=find_first_digit):
    model_id = model_config["name"]
    if model_config["access"] == "gated":
        hf_token = access_config["HF_TOKEN"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            quantization_config=bnb_config,
            token=hf_token
        )
    elif model_config["access"] == "remote_code":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            quantization_config=bnb_config
        )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
    )
    annotator = CasualLMClassifier(
        pipeline,
        label_extraction_function=label_extraction_function,
        temperature=model_config["temperature"],
        do_sample=model_config["do_sample"]
    )
    return annotator


def build_casual_lm_classifier_from_json_files(
        model_config_path,
        access_config_path,
        label_extraction_function=find_first_digit
):
    model_config = json.load(open(model_config_path, encoding='utf-8'))
    access_config = json.load(open(access_config_path, encoding='utf-8'))
    annotator = build_casual_lm_classifier(model_config, access_config, label_extraction_function)
    return annotator


def build_llama3_classifier():
    return build_casual_lm_classifier_from_json_files(llama3_config_path, access_config_path)


def build_llama2_classifier():
    return build_casual_lm_classifier_from_json_files(llama2_config_path, access_config_path)


def build_mistral_classifier():
    return build_casual_lm_classifier_from_json_files(mistral_config_path, access_config_path)


def build_mpt_classifier():
    return build_casual_lm_classifier_from_json_files(mpt_config_path, access_config_path)


def build_vicuna_classifier():
    return build_casual_lm_classifier_from_json_files(vicuna_config_path, access_config_path)
