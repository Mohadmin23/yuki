"""
Yuki fine-tune: Qwen3-32B + QLoRA via Unsloth.

Runs on the Thunder Compute A100/H100 (80GB). Do not run on the M1 Air.

Usage on the remote box:
    cd /home/ubuntu/yuki-finetune
    python train.py \
        --dataset /home/ubuntu/yuki-finetune/yuki_clean_v4.jsonl \
        --output /home/ubuntu/yuki-finetune/output \
        --epochs 3

Outputs:
    output/yuki-qwen3-32b-lora/   -- LoRA adapter (small, ~300MB)
    output/yuki-qwen3-32b-merged/ -- merged fp16 weights (~62GB, optional)
    output/yuki-qwen3-32b-gguf/   -- gguf quantized for llama.cpp (optional)
"""

from __future__ import annotations

# Unsloth must be imported before transformers/trl so its patches apply.
import unsloth  # noqa: F401

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


MODEL_NAME = "unsloth/Qwen3-32B-bnb-4bit"  # pre-quantized 4-bit, faster download
MAX_SEQ_LEN = 1024


def load_dataset(path: Path) -> Dataset:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_for_qwen(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-fp16", action="store_true", help="export merged fp16 weights")
    ap.add_argument("--export-gguf", type=str, default="", help="gguf quant level, e.g. q4_k_m")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output / "yuki-qwen3-32b-lora"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    ds = load_dataset(args.dataset)
    ds = ds.map(lambda ex: format_for_qwen(ex, tokenizer), remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(args.output / "checkpoints"),
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            packing=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            seed=args.seed,
        ),
    )

    trainer.train()

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved LoRA adapter to {adapter_dir}")

    if args.merge_fp16:
        merged_dir = args.output / "yuki-qwen3-32b-merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"saved merged fp16 to {merged_dir}")

    if args.export_gguf:
        gguf_dir = args.output / "yuki-qwen3-32b-gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=args.export_gguf)
        print(f"saved gguf ({args.export_gguf}) to {gguf_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
