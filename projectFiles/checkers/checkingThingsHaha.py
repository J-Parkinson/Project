import torch
from transformers import BertTokenizerFast, BertForMaskedLM

from projectFiles.seq2seq.constants import device

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                        output_hidden_states=True)
if torch.cuda.is_available():
    model.cuda()
model.eval()

sent = torch.tensor(tokenizer.encode("I tried to eat asparagus but unfortunately I contracted hi."), device=device)
print("Initial sentence:", sent)
enc = model.get_input_embeddings()(sent)
dec = model.get_output_embeddings()(enc)

print("Decoded sentence:", tokenizer.decode(dec.softmax(0).argmax(1)))
