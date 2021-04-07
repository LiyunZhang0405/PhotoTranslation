from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def load_model(path):
    return AutoModelForSeq2SeqLM.from_pretrained(path)

def translate(tokenizer, model, text):
    translated = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
