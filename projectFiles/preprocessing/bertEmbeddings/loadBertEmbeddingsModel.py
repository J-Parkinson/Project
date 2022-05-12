import torch
from transformers import BertTokenizerFast, BertForMaskedLM

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                        output_hidden_states=True)

if torch.cuda.is_available():
    model.cuda()
model.eval()

embeddingOutput = model.get_output_embeddings()
