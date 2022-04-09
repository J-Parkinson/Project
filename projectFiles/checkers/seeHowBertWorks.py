from transformers import BertTokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "The person who worte this cannot spel very welll and so made a lot of concurning mistakes when he wrote this."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print(tokenized_text)

print(tokenizer.convert_tokens_to_ids(tokenized_text))

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_text)))
