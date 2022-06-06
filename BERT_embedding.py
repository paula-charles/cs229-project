import torch
from transformers import BertTokenizer
from transformers import BertModel

import csv
import pandas as pd


# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)

def get_embedding(text):
    '''
    This function creates BERT tensor embeddings for a single title.
    '''
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    model.eval()

    # Run the text through BERT, get the output and collect all of the hidden states produced from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor)
        # can use last hidden state as word embeddings
        last_hidden_state = outputs[0]
        word_embed_1 = last_hidden_state

    embedding1 = word_embed_1[0]

    return embedding1


def main():

    df = pd.read_csv('ted_main.csv', delimiter=',')
    tuples = [list(x) for x in df.values]

    n_examples_total = len(tuples)
    n_examples = int(n_examples_total*0.7)
    n_val = int(n_examples_total*0.9)

    training_data = tuples[:n_examples]
    validation_data = tuples[n_examples:n_val]
    test_data = tuples[n_val:]

    names = [list(datapoint[7].rpartition(": "))[-1] for datapoint in training_data]
    number_comments = [datapoint[0] for datapoint in training_data]
    number_views = [datapoint[-1] for datapoint in training_data]

    embedded_names = []

    for text in names:
        embedded_names.append(get_embedding(text))


if __name__ == "__main__":
    main()
