import argparse
import json

import numpy as np
from BERT_model import TastyModel
from BERT_utils import CONFIG
from sklearn.model_selection import KFold
from utils import ENTITIES, prepare_data


def cross_validate(args):
    """
    Performs k-fold cross-validation training and evaluation of the TastyModel.

    Arguments:
        args: Parsed command-line arguments containing:
            - bert_type (str): BERT model type.
            - model_name_or_path (str): Path to a pretrained model checkpoint.
            - tasteset_path (str): Path to the TASTEset CSV file.
            - num_of_folds (int): Number of folds for cross-validation.
            - seed (int): Seed for reproducibility.
            - use_crf (bool): Whether to use CRF on top of BERT.

    Returns:
        None. Results are printed and saved to a JSON file.
    """
    bio_ingredients, bio_entities = prepare_data(args.tasteset_path, "bio")

    CONFIG["bert_type"] = args.bert_type
    CONFIG["model_name_or_path"] = args.model_name_or_path
    CONFIG["training_args"]["output_dir"] = f"./models/{args.bert_type}"
    CONFIG["use_crf"] = args.use_crf
    CONFIG["training_args"]["seed"] = args.seed

    kf = KFold(n_splits=args.num_of_folds, shuffle=True, random_state=args.seed)
    cross_val_results = {}

    for fold_id, (train_index, test_index) in enumerate(kf.split(bio_entities)):
        tr_ingredients, vl_ingredients = [
            bio_ingredients[idx] for idx in train_index
        ], [bio_ingredients[idx] for idx in test_index]
        tr_entities, vl_entities = [bio_entities[idx] for idx in train_index], [
            bio_entities[idx] for idx in test_index
        ]

        model = TastyModel(config=CONFIG)
        model.train(tr_ingredients, tr_entities)
        results = model.evaluate(vl_ingredients, vl_entities)
        print(results)
        cross_val_results[fold_id] = results

    sub_res_folder = args.bert_type.split("/")[-1]
    with open(f"./res/{sub_res_folder}_cross_val_results.json", "w") as json_file:
        json.dump(cross_val_results, json_file, indent=4)

    # aggregate and print results
    cross_val_results_aggregated = {
        entity: {"precision": [], "recall": [], "f1": []}
        for entity in ENTITIES + ["all"]
    }

    print(f"{'entity':^20s}{'precision':^15s}{'recall':^15s}{'f1-score':^15s}")
    for entity in cross_val_results_aggregated.keys():
        print(f"{entity:^20s}", end="")
        for metric in cross_val_results_aggregated[entity].keys():
            for fold_id in range(args.num_of_folds):
                cross_val_results_aggregated[entity][metric].append(
                    cross_val_results[fold_id][entity][metric]
                )

            mean = np.mean(cross_val_results_aggregated[entity][metric])
            mean = int(mean * 1000) / 1000
            std = np.std(cross_val_results_aggregated[entity][metric])
            std = int(std * 1000) / 1000 + 0.001 * round(std - int(std * 1000) / 1000)
            print(f"{mean:^2.3f} +- {std:^2.3f} ", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--bert-type", type=str, required=True, help="BERT type")
    parser.add_argument(
        "--model-name-or-path", type=str, help="path to model checkpoint"
    )
    parser.add_argument(
        "--tasteset-path",
        type=str,
        default="./data/TASTEset.csv",
        help="path to TASTEset",
    )
    parser.add_argument(
        "--num-of-folds",
        type=int,
        default=5,
        help="Number of folds in cross-validation",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Use CRF layer on top of BERT + linear layer",
    )
    args = parser.parse_args()

    cross_validate(args)
