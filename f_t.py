from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import numpy as np
# from nltk.metrics import masi_distance

# 1. Load your CSV summarization dataset
data_files = {
    "train": "train.csv",
    "validation": "validation.csv"
}
dataset = load_dataset('csv', data_files=data_files)


model_ckpt = "facebook/bart-base"  
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# 4. Preprocessing: tokenize inputs and targets
def preprocess(batch):
    
    model_inputs = tokenizer(
        batch["article"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["highlights"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 5. Apply tokenization to all splits
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names  # Keeps only processed tensors
)

# 6. Data collator for seq2seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_ckpt)

# 7. Metrics (ROUGE is typical for summarization)
from evaluate import load as load_metric
rouge = load_metric("rouge")

def compute_metrics(pred):
    preds, labels = pred.predictions, pred.label_ids
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: v.mid.fmeasure for k, v in result.items()}

# 8. Training arguments (tune for your resources)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=3e-5,
    save_total_limit=2,
    predict_with_generate=True,  # Needed for proper ROUGE, seq2seq output
    logging_steps=50,
    fp16=True,
    load_best_model_at_end=True
)

# 9. Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

# 10. Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

def get_summary(text, max_length=150):
    prompt = f"### Instruction:\nSummarize the following scientific article into a concise abstract.\n\n### Article:\n{text}\n\n### Summary:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
