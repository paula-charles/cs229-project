import torch
from transformers import BertTokenizer
from transformers import BertModel

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
