
import torch
from transformers import BertTokenizerFast, BertForMaskedLM

from projectFiles.constants import device

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                        output_hidden_states=True)
if torch.cuda.is_available():
    model.cuda()
model.eval()

sent1 = torch.tensor(
    tokenizer.encode("I tried to eat asparagus but unfortunately I contracted hideosyncraticosteoperosis. [PAD]"),
    device=device)
sent2 = torch.tensor(
    tokenizer.encode("Unfortunately in trying to eat my parsley I also farted up two rabid ferrets. [PAD]"),
    device=device)
sent3 = torch.tensor([0 for _ in range(len(sent1))], device=device)
print(sent3)
sents = torch.vstack((sent1, sent2, sent3))
enc = model.get_input_embeddings()(sents)
dec = model.get_output_embeddings()(enc)

print(dec)
print(dec[0].softmax(0).argmax(1).cpu().numpy().tolist())

print("Decoded sentence:", tokenizer.decode(dec[0].softmax(0).argmax(1)))
print("Decoded sentence:", tokenizer.decode(dec[2].softmax(0).argmax(1)))
