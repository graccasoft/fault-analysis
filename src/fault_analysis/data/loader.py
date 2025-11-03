from typing import Iterable, List, Literal, Type, Any
import orjson
from datasets import Dataset
from .schema import InstructionRecord, FaultRecord


SchemaName = Literal["instruction", "fault"]


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)


def load_records(path: str, schema: SchemaName) -> List[Any]:
    model: Type[Any] = InstructionRecord if schema == "instruction" else FaultRecord
    return [model(**obj) for obj in _iter_jsonl(path)]


def to_hf_dataset(objs: List[Any]) -> Dataset:
    dicts = [o.model_dump() for o in objs]
    return Dataset.from_list(dicts)
