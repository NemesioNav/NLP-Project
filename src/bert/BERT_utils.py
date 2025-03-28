import re

import spacy

NLP = spacy.load("en_core_web_sm")

ENTITIES = [
    "FOOD",
    "QUANTITY",
    "UNIT",
    "PROCESS",
    "PHYSICAL_QUALITY",
    "COLOR",
    "TASTE",
    "PURPOSE",
    "PART",
    "TRADE_NAME",
    "DIET",
    "EXAMPLE",
]
NEWLINE_CHAR = "."

LABEL2ID = {"O": 0}
idx = 1
for entity in ENTITIES:
    LABEL2ID[f"B-{entity}"] = idx
    idx += 1
    LABEL2ID[f"I-{entity}"] = idx
    idx += 1

CONFIG = {
    "bert_type": None,
    "model_name_or_path": None,
    "num_of_tokens": 128,
    "only_first_token": True,
    "training_args": {
        "output_dir": None,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 30,
        "weight_decay": 0.01,
    },
    "label2id": LABEL2ID,
}


def check_if_entity_correctly_began(entity, prev_entity):
    """
    Checks if "I-" entity is preceded with "B-" or "I-" of the same entity type.

    Arguments:
        entity (str): Current entity label
        prev_entity (str): Previous entity label

    Returns:
        bool: True if the entity is correctly started, False otherwise
    """
    if "I-" in entity and re.sub(r"[BI]-", "", entity) != re.sub(
        r"[BI]-", "", prev_entity
    ):
        return False
    return True


def token_to_entity_predictions(
    text_split_words, text_split_tokens, token_labels, id2label
):
    """
    Transforms token (subword) predictions into word predictions.

    Arguments:
        text_split_words (list): List of words from one recipe ingredients,
            e.g. ["2", "carrots"] (the ones that go to tokenizer)
        text_split_tokens (list): List of tokens from one recipe ingredients,
            e.g. ["2", "car", "##rots"] (the ones that arise from input decoding)
        token_labels (list): List of labels associated with each token from
            text_split_tokens
        id2label (dict): A mapping from ids (0, 1, ...) to labels ("B-FOOD",
            "I-FOOD", ...)

    Returns:
        list: A list of entities associated with each word from text_split_words,
            i.e. entities extracted from recipe ingredients
    """

    word_idx = 0
    word_entities = []
    word_from_tokens = ""
    word_entity = ""
    prev_word_entity = ""

    for token_label, token in zip(token_labels, text_split_tokens):
        if token in ["[SEP]", "[CLS]"]:
            continue
        word_from_tokens += re.sub(r"^##", "", token)
        # take the entity associated with the first token (subword)
        word_entity = id2label[token_label] if word_entity == "" else word_entity

        if (
            word_from_tokens == text_split_words[word_idx]
            or word_from_tokens == "[UNK]"
        ):
            word_idx += 1
            # replace entities containing "I-" that do not have a predecessor
            # with "B-"
            word_entity = (
                "O"
                if not check_if_entity_correctly_began(word_entity, prev_word_entity)
                else word_entity
            )
            word_entities.append(word_entity)
            word_from_tokens = ""
            prev_word_entity = word_entity
            word_entity = ""

    return word_entities


def tokenize_and_align_labels(
    ingredients, entities, tokenizer, max_length, label2id, only_first_token=True
):
    """
    Tokenizes ingredients and aligns entity labels with tokens.

    Arguments:
        ingredients (list): List of lists of words from recipe ingredients
        entities (list): List of lists of entities from recipe ingredients
        tokenizer: Tokenizer to use for tokenization
        max_length (int): Maximal tokenization length
        label2id (dict): A mapping from labels ("B-FOOD", "I-FOOD", ...) to ids
            (0, 1, ...)
        only_first_token (bool): Whether to label only first subword of a word.
            E.g. if "chicken" is split into "chic", "##ken", then if True, it will
            have [1, -100], if False [1, 1]. -100 is omitted in Pytorch loss function

    Returns:
        dict: A dictionary with tokenized recipes ingredients with/without
            associated token labels
    """
    tokenized_data = tokenizer(
        ingredients, truncation=True, max_length=max_length, is_split_into_words=True
    )

    labels = []
    ingredients_words_beginnings = []  # mark all first subwords,
    # e.g. 'white sugar' which is split into ["wh", "##ite", "sug". "##ar"]
    # would have [1, 0, 1, 0]. This is used as prediction mask in the BertCRF

    for recipe_idx in range(len(ingredients)):
        # Map tokens to their respective word.
        word_ids = tokenized_data.word_ids(batch_index=recipe_idx)
        previous_word_idx = None
        label_ids = []
        words_beginnings = []
        for word_idx in word_ids:
            if word_idx is None:
                words_beginnings.append(False)
            elif word_idx != previous_word_idx:
                words_beginnings.append(True)
            else:
                words_beginnings.append(False)
            if entities:
                if word_idx is None:
                    new_label = -100
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    new_label = label2id[entities[recipe_idx][word_idx]]
                else:
                    new_label = (
                        -100
                        if only_first_token
                        else label2id[entities[recipe_idx][word_idx]]
                    )
                label_ids.append(new_label)
            previous_word_idx = word_idx

        words_beginnings += (max_length - len(words_beginnings)) * [False]
        ingredients_words_beginnings.append(words_beginnings)
        if entities:
            labels.append(label_ids)

    if entities:
        tokenized_data["labels"] = labels

    tokenized_data["ingredients"] = ingredients
    tokenized_data["prediction_mask"] = ingredients_words_beginnings

    return tokenized_data


def tokenize_ingredients(ingredients):
    """
    Tokenizes a string of ingredients using spaCy.

    Arguments:
        ingredients (str): String of ingredients

    Returns:
        list: List of tokenized ingredients
    """
    doc = NLP(ingredients)
    tokenized_ingredients = [token.text for token in doc]
    return tokenized_ingredients


def prepare_ingredients_for_prediction(ingredients):
    """
    Prepares ingredients for entity extraction, handling different input formats.

    Arguments:
        ingredients: Input can be:
            * one string of ingredients (str): "2 carrots..."
            * multiple lists of ingredients (list -> str): ["2 carrots...",
              "one tablespoon of sugar..."] (each element is from a different recipe)
            * multiple lists of tokenized ingredients (list -> list -> str):
              [["2", "carrots", ...], ["one", "tablespoon", "of", "sugar"], ...]
              (each element is from a different recipe)

    Returns:
        list: List of lists of tokenized ingredients with newlines replaced by NEWLINE_CHAR
    """
    if isinstance(ingredients, list):
        if isinstance(ingredients[0], str):  # list of ingredients
            ingredients = [tokenize_ingredients(ingr) for ingr in ingredients]
        elif isinstance(ingredients[0], list):  # list of tokenized ingredients
            ingredients = ingredients
        else:
            raise ValueError(f"{type(ingredients[0])} is not supported!")

    elif isinstance(ingredients, str):
        ingredients = [tokenize_ingredients(ingredients)]

    else:
        raise ValueError(f"{type(ingredients)} is not supported!")

    ingredients = [
        [NEWLINE_CHAR if token == "\n" else token for token in ingreds]
        for ingreds in ingredients
    ]
    return ingredients
