import json
import os
import sys

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PATH not in sys.path:
    sys.path.append(PATH)

import bert.BERT_model as BERT_model
from llm.helper import FewShotRetriever, LLMEntityExtractor

def inference():
    # Load data

    df = pd.read_csv('/home/ids/nnavarro-24/NLP-Project/data/TASTEset.csv')

    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    ingredients_train = df_train['ingredients'].str.replace('\n', '. ').to_list()
    ingredients_test = df_test['ingredients'].str.replace('\n', '. ').to_list()

    _, entities_token_train = BERT_model.prepare_data(df_train, "bio")
    _, entities_token_test = BERT_model.prepare_data(df_test, "bio")

    train_data = {'ingredients': ingredients_train, 'entities': entities_token_train}
    test_data = {'ingredients': ingredients_test, 'entities': entities_token_test}

    train_data = datasets.Dataset.from_dict(train_data)
    test_data = datasets.Dataset.from_dict(test_data)

    data = {'train': train_data, 'test': test_data}

    # Build retriever on training data
    retriever = FewShotRetriever(data['train'])
    retriever.build_index()

    # Build LLM entity extractor
    llm_extractor = LLMEntityExtractor("meta-llama/Llama-3.2-3B", retriever)

    results = {}
    # Extract entities from test data
    for n_few_shot in [1, 2, 4, 8]:
        print('=== {} ==='.format(n_few_shot))
        llm_extractor.n_few_shot=n_few_shot
        metrics = llm_extractor.eval(data['test'], variable_size=True)
        print(json.dumps(metrics, indent=2))
        results[n_few_shot] = metrics

    # Save results to a JSON file
    with open('/home/ids/nnavarro-24/NLP-Project/results/llm.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    inference()