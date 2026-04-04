import os
from pathlib import Path

workspace_hint = os.environ.get("READINESS_WORKING_DIR")
WORKSPACE = Path(workspace_hint).resolve() if workspace_hint else Path.cwd().resolve()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if not any(os.environ.get(name) for name in ("HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HOME")):
    os.environ["HF_HOME"] = str(WORKSPACE / "huggingface-cache")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DATASET_REPO_ID = "karthiksagarn/astro_horoscope"
DATASET_SPLIT = "train"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
train_source = load_dataset(DATASET_REPO_ID, split=DATASET_SPLIT)


def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )


tokenized = train_source.map(tokenize, batched=True, remove_columns=train_source.column_names)
tokenized = tokenized.train_test_split(test_size=0.1)

model = AutoModelForCausalLM.from_pretrained(MODEL_REPO_ID, dtype="auto")
model = model.to("npu")

training_args = TrainingArguments(
    output_dir=str(WORKSPACE / "qwen3-finetuned"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
