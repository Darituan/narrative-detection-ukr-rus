import random
import json

import pandas as pd


generation_config_path = 'config/generation_config.json'
narrative_data_path = 'narratives.json'


def generate_narrative_data(generator, system_prompt, user_prompt, narrative_label, language_label, n_texts):
    content = []
    if type(user_prompt) == list:
        for _ in range(n_texts):
            prompt = random.choice(user_prompt)
            text = generator.generate_narrative_text(prompt, system_prompt)
            content.append(text)
    else:
        for _ in range(n_texts):
            text = generator.generate_narrative_text(user_prompt, system_prompt)
            content.append(text)
    language = [language_label for _ in range(len(content))]
    narrative = [narrative_label for _ in range(len(content))]
    data = {'Language': language, 'Narrative': narrative, 'Content': content}
    dataframe = pd.DataFrame.from_dict(data)
    return dataframe


def generate_narrative_dataset(generator, system_prompts, user_prompts, narrative_labels, language_labels, n_texts):
    dataframes = [
        generate_narrative_data(
            generator,
            system_prompts[i],
            user_prompts[i],
            narrative_labels[i],
            language_labels[i],
            n_texts[i]
        ) for i in range(len(narrative_labels))
    ]
    dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
    dataframe = dataframe.sample(frac=1)
    return dataframe


def generate_narrative_dataset_from_config(generator):
    config = json.load(open(generation_config_path, encoding='utf-8'))
    narrative_data = json.load(open(narrative_data_path, encoding='utf-8'))
    narrative_labels = config["narrative_labels"]
    language_labels = config["language_labels"]
    n_texts = config["n_texts"]

    system_prompts = []
    user_prompts = []
    for i in range(len(narrative_labels)):
        prompt_data = narrative_data[narrative_labels[i]][language_labels[i]]["prompt"]
        system_prompts.append(prompt_data["system"])
        user_prompts.append(prompt_data["user"])

    return generate_narrative_dataset(generator, system_prompts, user_prompts, narrative_labels, language_labels, n_texts)


def generate_narrative_data_csv(generator, output_filename, system_prompt, user_prompt, narrative_label, language_label, n_texts):
    dataframe = generate_narrative_data(generator, system_prompt, user_prompt, narrative_label, language_label, n_texts)
    dataframe.to_csv(output_filename, index=False)


def generate_narrative_dataset_csv_from_config(generator, output_filename):
    dataframe = generate_narrative_dataset_from_config(generator)
    dataframe.to_csv(output_filename, index=False)
