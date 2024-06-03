import json

import pandas as pd

from prompt import format_prompt


narrative_data_path = 'narratives.json'
prompt_template_path = 'prompt_templates.json'
annotation_config_path = 'config/annotation_config.json'


def annotate_narratives(
        classifier,
        texts,
        narratives,
        narrative_ids,
        narrative_labels,
        prompt_template):
    prompts = list(map(lambda x: format_prompt(prompt_template, x, narratives, narrative_labels), texts))
    ids, responses = classifier.classify(prompts, narrative_ids)
    return ids, responses


def annotate_dataframe_casual_lm(
        classifier,
        dataframe,
        narrative_ids,
        narrative_labels,
        prompt_lang='ENG',
        prompt_id=0,
        narrative_langs=('UKR', 'UKR', 'UKR', 'ENG'),
        content_col_name='Content',
        write_responses=False
):
    narrative_ids_dict = {narrative_labels[i]: narrative_ids[i] for i in range(len(narrative_ids))}

    all_narratives = json.load(open(narrative_data_path, encoding='utf-8'))
    narratives = [all_narratives[narrative_ids[i]][narrative_langs[i]]['text'] for i in range(len(narrative_ids))]

    prompt_template_str = json.load(open(prompt_template_path, encoding='utf-8'))[prompt_id][prompt_lang]
    if type(prompt_template_str) == list:
        prompt_template_str = classifier.pipeline.tokenizer.apply_chat_template(
            prompt_template_str,
            tokenize=False,
            add_generation_prompt=True
        )

    texts = dataframe[content_col_name]
    ids, responses = annotate_narratives(
        classifier,
        texts,
        narratives,
        narrative_ids_dict,
        narrative_labels,
        prompt_template_str
    )

    dataframe['Narrative'] = ids
    if write_responses:
        dataframe['Response'] = responses

    return dataframe


def annotate_dataframe_from_config(classifier, dataframe):
    config = json.load(open(annotation_config_path, encoding='utf-8'))
    return annotate_dataframe_casual_lm(
        classifier,
        dataframe,
        config["narrative_ids"],
        config["narrative_labels"],
        config["prompt_lang"],
        config["prompt_id"],
        config["narrative_langs"],
        config["content_col_name"],
        config["write_responses"]
    )


def annotate_csv_from_config(classifier, input_csv_path, chunk_size=None, output_csv_path=None):
    if chunk_size is None:
        df = pd.read_csv(input_csv_path, header=0)
        result = annotate_dataframe_from_config(classifier, df)
    else:
        results = []
        for chunk in pd.read_csv(input_csv_path, header=0, chunksize=chunk_size):
            results.append(annotate_dataframe_from_config(classifier, chunk))
        result = pd.concat(results, ignore_index=True, sort=False)
    if output_csv_path is not None:
        result.to_csv(output_csv_path, index=False)
    return result
