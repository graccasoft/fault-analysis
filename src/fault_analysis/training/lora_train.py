from typing import Dict
import os
import yaml
from datasets import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from ..config import AppConfig
from ..data.loader import load_records, to_hf_dataset
from ..preprocess.clean import normalize_text


def _load_config(config_path: str) -> AppConfig:
    with open(config_path, "r") as f:
        d = yaml.safe_load(f)
    return AppConfig(**d)


def _build_datasets(cfg: AppConfig) -> DatasetDict:
    train = load_records(cfg.paths.train_path, cfg.schema)
    ds_train = to_hf_dataset(train)

    val = None
    if cfg.paths.val_path and os.path.exists(cfg.paths.val_path):
        val = load_records(cfg.paths.val_path, cfg.schema)
        ds_val = to_hf_dataset(val)
    else:
        ds_val = None

    ds = DatasetDict({"train": ds_train})
    if ds_val is not None:
        ds["validation"] = ds_val
    return ds


def _format_examples(ex: Dict, schema: str) -> Dict:
    if schema == "instruction":
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        src = f"Instruction: {instruction}\nInput: {normalize_text(inp)}\nAnswer:"
        tgt = ex.get("output", "")
        return {"source": src, "target": tgt}
    else:
        text = ex.get("text", "")
        q = "Classify the fault and suggest actions."
        src = f"Instruction: {q}\nInput: {normalize_text(text)}\nAnswer:"
        tgt = (ex.get("fault_type") or "")
        return {"source": src, "target": tgt}


def train(config_path: str) -> None:
    cfg = _load_config(config_path)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.train.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.train.model_name)

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, peft_cfg)

    ds = _build_datasets(cfg)

    def preprocess(batch):
        # batch is a dict of lists, need to convert to list of dicts
        # Get the first key to determine batch size
        first_key = list(batch.keys())[0]
        batch_size = len(batch[first_key])
        
        # Convert columnar format to row format
        examples = [
            {key: batch[key][i] for key in batch.keys()}
            for i in range(batch_size)
        ]
        formatted = [_format_examples(ex, cfg.schema) for ex in examples]
        model_inputs = tok(
            [f["source"] for f in formatted],
            max_length=cfg.train.max_source_length,
            truncation=True,
            padding="max_length",
        )
        labels = tok(
            [f["target"] for f in formatted],
            max_length=cfg.train.max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.paths.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        predict_with_generate=True,
        seed=cfg.train.seed,
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.paths.output_dir)
    tok.save_pretrained(cfg.paths.output_dir)
