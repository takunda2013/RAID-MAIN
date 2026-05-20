from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
tokenizer.save_pretrained("models/roberta_contrastive/tokenizer")