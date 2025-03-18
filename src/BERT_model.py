import argparse
import os

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from BERT_utils import (
    CONFIG,
    prepare_ingredients_for_prediction,
    token_to_entity_predictions,
    tokenize_and_align_labels,
)
from utils import evaluate_predictions, prepare_data


class TastyModel:
    """
    A class for fine-tuning and evaluating a BERT-based model for token classification.

    Attributes:
        config (dict): Configuration dictionary for the model.
        tokenizer: Tokenizer for the model.
        trainer: Trainer object for training and evaluation.
    """

    def __init__(self, config):
        self.config = config
        bert_type = self.config["bert_type"]
        model_name_or_path = (
            self.config["model_name_or_path"]
            if self.config["model_name_or_path"] is not None
            else bert_type
        )

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        label2id = {k: int(v) for k, v in self.config["label2id"].items()}
        id2label = {v: k for k, v in label2id.items()}

        # for reproducibility
        set_seed(self.config["training_args"]["seed"])

        # to discard pretrained classification layer
        ignore_mismatched_sizes = (
            True if self.config["model_name_or_path"] is not None else False
        )

        model_class = AutoModelForTokenClassification
        # model_class = BertForTokenClassification

        model = model_class.from_pretrained(
            model_name_or_path,
            num_labels=len(self.config["label2id"]),
            ignore_mismatched_sizes=(
                ignore_mismatched_sizes if "dslim" in bert_type else True
            ),
            label2id=label2id,
            id2label=id2label,
            classifier_dropout=0.2,
        )

        training_args = TrainingArguments(**self.config["training_args"])

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            max_length=self.config["num_of_tokens"],
            padding="max_length",
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def train(self, train_ingredients, train_entities):
        """
        Trains the model on the provided data.

        Arguments:
            train_ingredients (list): List of tokenized ingredients.
            train_entities (list): List of corresponding BIO entities.
        """
        _, train_dataset = self.prepare_data(train_ingredients, train_entities)

        self.trainer.train_dataset = train_dataset

        self.trainer.train()

    def evaluate(self, ingredients, entities):
        """
        Evaluates the model on the provided data.

        Arguments:
            ingredients (list): List of tokenized ingredients.
            entities (list): List of corresponding BIO entities.

        Returns:
            dict: Evaluation metrics.
        """
        pred_entities = self.predict(ingredients)

        results = evaluate_predictions(entities, pred_entities, "bio")

        return results

    def predict(self, ingredients):
        """
        Predicts entities for the given ingredients.

        Arguments:
            ingredients (list): List of tokenized ingredients.

        Returns:
            list: Predicted BIO entities.
        """
        ingredients = prepare_ingredients_for_prediction(ingredients)
        data, dataset = self.prepare_data(ingredients, [])
        preds = self.trainer.predict(dataset)

        if self.config["use_crf"] is True:
            token_labels = preds[0][1]
        else:
            token_probs = preds[0]
            token_labels = token_probs.argmax(axis=2)

        pred_entities = []

        num_of_recipes = dataset.num_rows

        for recipe_idx in range(num_of_recipes):
            text_split_words = ingredients[recipe_idx]
            text_split_tokens = self.tokenizer.convert_ids_to_tokens(
                data["input_ids"][recipe_idx]
            )

            id2label = self.trainer.model.config.id2label
            if self.config["use_crf"] is True:  # labels are associated to
                # first subwords, hence, are already the word entities
                word_entities = [
                    self.trainer.model.config.id2label[word_label]
                    for word_label in token_labels[recipe_idx]
                    if word_label != -100
                ]
            else:
                word_entities = token_to_entity_predictions(
                    text_split_words,
                    text_split_tokens,
                    token_labels[recipe_idx],
                    id2label,
                )
            pred_entities.append(word_entities)

        return pred_entities

    def prepare_data(self, ingredients, entities):
        """
        Prepares data for training or evaluation.

        Arguments:
            ingredients (list): List of tokenized ingredients.
            entities (list): List of corresponding BIO entities.

        Returns:
            tuple: Tokenized data and dataset.
        """
        data = tokenize_and_align_labels(
            ingredients=ingredients,
            entities=entities,
            tokenizer=self.tokenizer,
            label2id=self.trainer.model.config.label2id,
            max_length=self.config["num_of_tokens"],
            only_first_token=self.config["only_first_token"],
        )

        dataset = Dataset.from_dict(data)

        return data, dataset

    def save_model(self):
        """
        Saves the trained model and its configuration.
        """
        save_dir = (
            self.config["save_dir"] if self.config["save_dir"] else "taisti_ner_model"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Add custom config values to the config.json
        self.trainer.model.config.num_of_tokens = self.config["num_of_tokens"]
        self.trainer.model.config.only_first_token = self.config["only_first_token"]
        self.trainer.model.config.training_args = self.config["training_args"]
        self.trainer.model.config.model_pretrained_path = "."
        self.trainer.model.config.use_crf = self.config["use_crf"]

        self.trainer.save_model(save_dir)

        print(f"Model with configs saved in {os.path.abspath(save_dir)}!!!")


def train(args):
    """
    Trains the TastyModel using the provided arguments.

    Arguments:
        args: Command-line arguments.
    """
    CONFIG["bert_type"] = args.bert_type
    CONFIG["model_name_or_path"] = args.model_name_or_path
    CONFIG["use_crf"] = args.use_crf
    CONFIG["training_args"]["seed"] = args.seed
    CONFIG["save_dir"] = args.save_dir

    bio_ingredients, bio_entities = prepare_data(args.tasteset_path, "bio")

    model = TastyModel(config=CONFIG)
    model.train(bio_ingredients, bio_entities)
    model.save_model()
