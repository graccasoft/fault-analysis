from typing import Optional
from pydantic import BaseModel


class Paths(BaseModel):
    train_path: str
    val_path: Optional[str] = None
    output_dir: str = "outputs/model"


class LoraConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[list[str]] = None


class TrainConfig(BaseModel):
    model_name: str = "google/flan-t5-base"
    max_source_length: int = 512
    max_target_length: int = 256
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 2e-4
    num_train_epochs: float = 10.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    seed: int = 42


class AppConfig(BaseModel):
    schema: str = "instruction"
    paths: Paths
    lora: LoraConfig = LoraConfig()
    train: TrainConfig = TrainConfig()
