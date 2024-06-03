import pandas as pd


def filter_df_by_consensus(dataframes, label_column, threshold=0.9):

    df = pd.concat([df[label_column] for df in dataframes], axis=1)

    most_common_values = []
    for index, row in df.iterrows():

        most_common_value = row.mode().values[0]
        most_common_count = row.value_counts()[most_common_value]

        if most_common_count / len(row) < threshold:
            most_common_value = -1

        most_common_values.append(most_common_value)

    df1 = dataframes[0]
    df1[label_column] = most_common_values
    df1 = df1[df1[label_column] >= 0]
    return df1


def filter_csv_by_consensus(annotated_csv_files, output_filename, label_column='Narrative', threshold=0.9, write_index=False):
    dataframes = [pd.read_csv(file, header=0) for file in annotated_csv_files]
    df = filter_df_by_consensus(dataframes, label_column, threshold)
    df.to_csv(output_filename, index=write_index)


if __name__ == '__main__':
    df = pd.read_csv('test_4nar_consensus_60.csv', header=0)
    print(df.shape)
    df1 = pd.read_csv('test_4nar_consensus_90.csv', header=0)
    print(df1.shape)
    # models = ['llama3', 'llama2', 'mistral']
    # files = [f'test_4nar_{model}.csv' for model in models]
    # out = 'test_4nar_consensus_60.csv'
    # filter_csv_by_consensus(files, out, threshold=0.6)
