import openai
import anthropic

import json
from abc import ABC, abstractmethod


access_config_path = 'config/access_config.json'
gpt_4o_config_path = 'config/models/gpt_4o_config.json'
claude_opus_config_path = 'config/models/claude_opus_config.json'

openai_key = json.load(open(access_config_path, encoding='utf-8'))["openai_key"]
anthropic_key = json.load(open(access_config_path, encoding='utf-8'))["claude_key"]


class NarrativeTextGenerator(ABC):
    @abstractmethod
    def generate_narrative_text(self, user_prompt, system_prompt):
        pass


class OpenAITextGenerator(NarrativeTextGenerator):
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens

    def generate_narrative_text(self, user_prompt, system_prompt):
        client = openai.OpenAI(
            api_key=openai_key
        )

        completion = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": user_prompt}
            ]
        )

        return completion.choices[0].message.content


class AnthropicTextGenerator(NarrativeTextGenerator):
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens

    def generate_narrative_text(self, user_prompt, system_prompt):
        client = anthropic.Anthropic(
            api_key=anthropic_key,
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return message.content


def build_openai_text_generator(config):
    return OpenAITextGenerator(config["model"], config["max_tokens"])


def build_anthropic_text_generator(config):
    return AnthropicTextGenerator(config["model"], config["max_tokens"])


def build_openai_text_generator_from_file(config_path):
    config = json.load(open(config_path, encoding='utf-8'))
    return build_openai_text_generator(config)


def build_anthropic_text_generator_from_file(config_path):
    config = json.load(open(config_path, encoding='utf-8'))
    return build_anthropic_text_generator(config)


def build_gpt_4o_text_generator():
    return build_anthropic_text_generator_from_file(gpt_4o_config_path)


def build_claude_opus_text_generator():
    return build_anthropic_text_generator_from_file(claude_opus_config_path)
