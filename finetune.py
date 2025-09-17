from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


dataset = load_dataset("ccdv/arxiv-summarization")

def format_example(example):
    instruction = "Summarize the following scientific article into a concise abstract."
    input_text = example["article"]
    output_text = example["abstract"]
    prompt = f"### Instruction:\n{instruction}\n\n### Article:\n{input_text}\n\n### Summary:\n"
    return {"prompt": prompt, "label": output_text}

dataset = dataset.map(format_example)


MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)


model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def tokenize(batch):
    inputs = tokenizer(
        batch["prompt"], max_length=2048, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["label"], max_length=512, truncation=True, padding="max_length"
    )


    inputs["labels"] = [
        [-100] * len(inp) + lab[len(inp):] if len(lab) > len(inp) else lab
        for inp, lab in zip(inputs["input_ids"], labels["input_ids"])
    ]
    return inputs

tokenized = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qwen_arxiv_summarization",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    fp16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("qwen-arxiv-summarizer-lora")

