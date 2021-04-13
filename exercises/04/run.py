import json

import papermill as pm


datasets = ['kc1', 'cm1']

for dataset in datasets:
    dataset_config_path=f"config/dataset/{dataset}.json"
    pm.execute_notebook(
        "main.ipynb",
        f"runs/dataset_{dataset}.ipynb",
        parameters = dict(
            DATASET_CONFIG_PATH=dataset_config_path,
            PROFILE_REPORTING=False,
        )
    )

