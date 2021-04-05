import csv
import json

import pandas as pd

experiments = []

repetition = 0

models = {}

with open('results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        configuration = row['configuration'].split('-')
        dataset_name, model_name = configuration[0], configuration[1]

        with open(f'config/dataset/{dataset_name.split("_")[1]}.json') as f:
            dataset_config = json.load(f)

        with open(f'config/model/{model_name.split("_")[1]}.json') as f:
            model_config = json.load(f)

        if repetition < 4:
            models[model_name] = {'rows': [row], 'dataset_config': dataset_config, 'model_config': model_config}
        else:
            models[model_name]['rows'].append(row)

        repetition += 1

        if repetition == 40:
            for model, info in models.items():
                df = pd.DataFrame(info['rows'])
                means = {
                    'true_pos': pd.to_numeric(df['true_pos']).mean(),
                    'false_pos': pd.to_numeric(df['false_pos']).mean(),
                    'false_neg': pd.to_numeric(df['false_neg']).mean(),
                    'true_neg': pd.to_numeric(df['true_neg']).mean(),
                    'precision': pd.to_numeric(df['precision']).mean(),
                    'sensitivity': pd.to_numeric(df['sensitivity']).mean(),
                    'f1': pd.to_numeric(df['f1']).mean(),
                }

                experiments.append(dict(**means, **info['dataset_config'], **info['model_config']))
                repetition = 0


df = pd.DataFrame(experiments)
with open('result_table.html', 'w') as f:
    f.write(df.sort_values(by='f1', ascending=False).reset_index(drop=True).to_html())

