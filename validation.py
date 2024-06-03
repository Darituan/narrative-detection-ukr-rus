import json
import random

import pandas as pd


def edit_annotated_df_manually(
        dataframe,
        narrative_dict,
        content_col="Content",
        label_col="Narrative",
        limiter=None,
        limit=None):
    rows = dataframe.shape[0]
    n = rows
    if limiter == "rows":
        n = limit
    elif limiter == "part":
        n = int(rows * limit)

    if n < rows:
        indices = random.sample(range(rows), n)
        dataframe = dataframe.loc[indices]

    narrative_options = "\nID, Narrative\n"
    for key, val in narrative_dict.items():
        narrative_options += f"{key}, {val}\n"

    i = 1
    correct_ids = []
    for index, row in dataframe.iterrows():
        narrative_id = input(f"\n\n{i})\nTEXT:\n{row[content_col]}\n\nWhich narrative does the text contain?\n\nOPTIONS:{narrative_options}\nGENERATED ANSWER (ID): {row[label_col]}\n\nEnter correct answer (ID): ")
        narrative_id = int(narrative_id)
        correct_ids.append(narrative_id)
        i += 1
    dataframe[label_col+'_correct'] = correct_ids
    return dataframe


def get_pos_neg_distribution(dataframe, labels, pred_col='Narrative', true_col='Narrative_correct'):
    res = {label: {} for label in labels}
    total_true_pos = 0
    total_true_neg = 0
    total_false_pos = 0
    total_false_neg = 0
    for label in labels:
        true_positives = dataframe[(dataframe[true_col] == label) & (dataframe[pred_col] == label)].shape[0]
        true_negatives = dataframe[(dataframe[true_col] != label) & (dataframe[pred_col] != label)].shape[0]
        false_positives = dataframe[(dataframe[true_col] != label) & (dataframe[pred_col] == label)].shape[0]
        false_negatives = dataframe[(dataframe[true_col] == label) & (dataframe[pred_col] != label)].shape[0]
        res[label]["true_pos"] = true_positives
        res[label]["true_neg"] = true_negatives
        res[label]["false_pos"] = false_positives
        res[label]["false_neg"] = false_negatives
        total_true_pos += true_positives
        total_true_neg += true_negatives
        total_false_pos += false_positives
        total_false_neg += false_negatives
    res["total"] = {"true_pos": total_true_pos, "true_neg": total_true_neg, "false_pos": total_false_pos, "false_neg": total_false_neg}
    return res


def get_precision_recall(pos_neg_distribution):
    res = {}
    avg_pre = 0
    avg_rec = 0
    n = 0
    for label, distribution in pos_neg_distribution.items():
        precision = distribution["true_pos"] / (distribution["true_pos"] + distribution["false_pos"] + 1e-8)
        recall = distribution["true_pos"] / (distribution["true_pos"] + distribution["false_neg"] + 1e-8)
        res[label] = {"precision": precision, "recall": recall}
        if label != "total":
            avg_pre += precision
            avg_rec += recall
            n += 1
    avg_pre /= n
    avg_rec /= n
    res["avg"] = {"precision": avg_pre, "recall": avg_rec}
    return res


def get_accuracy(pos_neg_distribution):
    scores = {}
    avg = 0
    n = 0
    for label, distribution in pos_neg_distribution.items():
        accuracy = (distribution["true_pos"] + distribution["true_neg"]) / (distribution["true_pos"] + distribution["true_neg"] + distribution["false_pos"] + distribution["false_neg"] + 1e-8)
        scores[label] = accuracy
        if label != "total":
            avg += accuracy
            n += 1
    avg /= n
    scores["avg"] = avg
    return scores


def get_f1_scores(precisions_recalls):
    scores = {}
    avg = 0
    n = 0
    for label, precision_recall in precisions_recalls.items():
        if label != "avg":
            f1 = 2 * precision_recall["precision"] * precision_recall["recall"] / (
                        precision_recall["precision"] + precision_recall["recall"] + 1e-8)
            scores[label] = f1
            if label != "total":
                avg += f1
                n += 1
    avg /= n
    scores["avg"] = avg
    return scores


def validate(dataframe, labels, pred_col='Narrative', true_col='Narrative_correct'):
    pos_neg_distribution = get_pos_neg_distribution(dataframe, labels, pred_col, true_col)
    precision_recall = get_precision_recall(pos_neg_distribution)
    f1 = get_f1_scores(precision_recall)
    accuracy = get_accuracy(pos_neg_distribution)

    res = {"pos_neg": pos_neg_distribution, "precision_recall": precision_recall, "f1": f1, "accuracy": accuracy}
    return res


if __name__ == '__main__':
    df = pd.read_csv('test_4nar_valid_60_mistral.csv', header=0, parse_dates=['Date'])
    narrative_ids = [5, 4, 6, 1, 0]

    valid = validate(df, narrative_ids)
    print(json.dumps(valid, indent=4))

