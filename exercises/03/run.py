from itertools import product
import json

from tqdm import tqdm
import papermill as pm


datasets = ['kc1', 'cm1']

dataset_config_template = {
    "dataset_path": "datasets/{}.csv",
    "delete_columns": [],
    "test_size": 0.0,
    "test_outlier_size": 0.0,
    "random_state": 0,
}

model_config_template = {
    "model_type": "",
    "hyperparameters": {}
}

models = [
    {'model_type': 'knndatadescription', 'hyperparameters': ['k', 'outlier_threshold']},
    {'model_type': 'oneclasssvm', 'hyperparameters': []},
]

dataset_configs = []
index = 0
for dataset in datasets:
    dataset_config = dataset_config_template.copy()
    dataset_config['dataset_path'] = dataset_config['dataset_path'].format(dataset)
    delete_columns = [[]]
    test_sizes = [0.4, 0.25]
    test_outlier_sizes = [0.0, 0.5, 0.6, 0.7]
    random_states = list(range(10))

    for delete_column, test_size, test_outlier_size, random_state in product(delete_columns, test_sizes, test_outlier_sizes, random_states):
    #for test in product(delete_columns, test_sizes, test_outlier_sizes, random_states):
        dataset_config['delete_columns'] = delete_column
        dataset_config['test_size'] = test_size
        dataset_config['test_outlier_size'] = test_outlier_size
        dataset_config['random_state'] = random_state

        index += 1
        dataset_configs.append((index, dataset_config.copy()))

model_configs = []
index = 0
for model in models:
    model_config = model_config_template.copy()
    model_config['model_type'] = model['model_type']
    if model_config['model_type'] == 'knndatadescription':
        ks = [1, 10, 100]
        outlier_thresholds = [1.0]
        for k, outlier_threshold in product(ks, outlier_thresholds):
            model_config['hyperparameters'] = {
                'k': k,
                'outlier_threshold': outlier_threshold,
            }

            index += 1
            model_configs.append((index, model_config.copy()))
    elif model_config['model_type'] == 'oneclasssvm':
        model_config['hyperparameters'] = {}

        index += 1
        model_configs.append((index, model_config.copy()))

for dataset_config, model_config in tqdm(list(product(dataset_configs, model_configs))):
    dataset_config_index = str(dataset_config[0]).zfill(6)
    model_config_index = str(model_config[0]).zfill(6)

    dataset_config_path=f"config/dataset/{dataset_config_index}.json"
    with open(dataset_config_path, 'w') as f:
        json.dump(dataset_config[1], f)

    model_config_path=f"config/model/{model_config_index}.json"
    with open(model_config_path, 'w') as f:
        json.dump(model_config[1], f)

    pm.execute_notebook(
        "main.ipynb",
        f"runs/dataset_{dataset_config_index}-model_{model_config_index}.ipynb",
        parameters = dict(
            DATASET_CONFIG_PATH=dataset_config_path,
            MODEL_CONFIG_PATH=model_config_path,
            PROFILE_REPORTING=False,
        )
    )

