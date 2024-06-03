import json

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from annotation import annotation_config_path


language_dict = {"UKR": Language.UKRAINIAN, "RUS": Language.RUSSIAN}

narrative_data_path = 'narratives.json'


def build_detector():
    languages = Language.all()
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return detector


def filter_dataset_by_lang(dataset, detector, language, confidence_threshold=0.6, content_col_name='Content'):
    # df = pd.read_csv(dataset_csv, header=header)
    mask = dataset.apply(
        lambda row: detector.compute_language_confidence(str(row[content_col_name]), language) > confidence_threshold,
        axis=1
    )
    df_lang = dataset[mask]
    return df_lang


def contains_words_or_phrases(text, words, phrases=[]):
    text = text.lower()
    for word in words:
        if word in text:
            return True
    for phrase in phrases:
        result = True
        for w in phrase:
            if w not in text:
                result = False
        if result:
            return True
    return False


def filter_dataset_by_keywords(dataset, keywords, key_phrases=[], content_col_name='Content'):
    mask = dataset.apply(
        lambda row: contains_words_or_phrases(str(row[content_col_name]), keywords, key_phrases),
        axis=1
    )
    filtered_df = dataset[mask]
    return filtered_df


def filter_dataset_by_keywords_neg(dataset, keywords, key_phrases=[], content_col_name='Content'):
    mask = dataset.apply(
        lambda row: not contains_words_or_phrases(str(row[content_col_name]), keywords, key_phrases),
        axis=1
    )
    filtered_df = dataset[mask]
    return filtered_df


def concat_csv(csv_files, output_filename):
    dataframes = [pd.read_csv(csv_file, header=0, parse_dates=['Date']) for csv_file in csv_files]
    result = pd.concat(dataframes, ignore_index=True, sort=False)
    result.to_csv(output_filename, index=False)


def filter_dataset_from_config(dataset):
    config = json.load(open(annotation_config_path, mode='r', encoding='utf-8'))
    narrative_data = json.load(open(narrative_data_path, mode='r', encoding='utf-8'))
    narrative_ids = config["narrative_ids"]
    narrative_langs = config["narrative_langs"]
    content_col_name = config["content_col_name"]

    n = len(narrative_ids)
    filtered_datasets = []
    for i in range(n):
        narrative = narrative_data[narrative_ids[i]][narrative_langs[i]]
        keywords = narrative["keywords"]
        key_phrases = narrative["key_phrases"]
        filtered = filter_dataset_by_keywords(dataset, keywords, key_phrases, content_col_name)
        filtered_datasets.append(filtered)

    result = pd.concat(filtered_datasets, ignore_index=True, sort=False)
    result.drop_duplicates(subset=[content_col_name], keep="first", inplace=True)
    return result


if __name__ == '__main__':
    # chunk_size = 500
    # i = 0
    # for chunk in pd.read_csv('data/test_4nar.csv', header=0, parse_dates=['Date'], chunksize=chunk_size):
    #     i += 1
    #     chunk.to_csv(f'data/test_4nar_{i}.csv', index=False)


    # df = pd.read_csv(f'data/mohil.csv', header=0, parse_dates=['Date'])
    # print(df.shape)


    # test_4nar.to_csv('data/test_4nar.csv', index=False)

    model_name = 'mistral'

    files = [f'data/test_4nar_{i+1}_{model_name}.csv' for i in range(6)]
    out = f'test_4nar_{model_name}.csv'

    concat_csv(files, out)

    # for i in range(9):
    #     df = pd.read_csv(f'data/rus_{i+1}.csv', header=0, parse_dates=['Date'])
    #
    #     narratives = json.load(open('narratives.json', mode='r', encoding='utf-8'))
    #     keywords = narratives[-3]['RUS']['keywords']
    #     key_phrases = narratives[-3]['RUS']['key_phrases']
    #
    #     filtered = filter_dataset_by_keywords(df, keywords, key_phrases)
    #     print(filtered.shape)
    #     filtered.to_csv(f'data/israel3_rus_{i+1}.csv', index=False)

    # df1 = pd.read_csv('data/nar0_ukr_small.csv', header=0, parse_dates=['Date'])

    # df2 = pd.read_csv('data/test_3nar_ukr_llama3.csv', header=0, parse_dates=['Date'])
    # print(df2.shape)
    #
    # mask = df2.apply(lambda row: row['Narrative'] == 2, axis=1)
    #
    # df = df2[mask]
    # print(df.shape)
    # print(df.iloc[0]['Content'])

    # df = pd.read_csv('data/rus_with_lang.csv', header=0, parse_dates=['Date'])
    # print(df.shape)
    # df.drop_duplicates(subset=['Content'], keep="first", inplace=True)
    # print(df.shape)
    # df.to_csv('data/rus_no_dup.csv', index=False)


    # df3 = pd.read_csv('data/nar2_ukr.csv', header=0, parse_dates=['Date'])
    #
    # df4 = pd.concat([df1, df2, df3], ignore_index=True, sort=False)
    #
    # df4.drop_duplicates(subset=['Content'], keep="first")
    #
    # df4.to_csv('data/test_3nar_ukr.csv', index=False)

    # narratives = json.load(open('narratives.json', mode='r', encoding='utf-8'))
    # keywords = narratives[2]['UKR']['keywords']
    # key_phrases = narratives[2]['UKR']['key_phrases']

    # print(keywords)
    # print(key_phrases)

    # nar1_ukr = filter_dataset_by_keywords(df, keywords)
    # print(nar1_ukr.shape)
    # nar1_ukr.to_csv('data/mohil.csv', index=False)
