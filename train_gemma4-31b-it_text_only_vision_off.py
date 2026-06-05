#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma-4-31B SFT — OOM-safe Unsloth script

Цель:
- тренировать Gemma-4-31B-it через Unsloth Core без Studio
- держать VRAM близко к Studio-профилю
- не ловить OOM на ровном месте
- проверить, что train_on_responses_only реально оставил assistant/model labels,
  а не сделал градиент почти нулевым из-за битой маски

Запуск:
    python train_gemma4-31b-it_text_only_vision_off.py

Можно переопределить:
    python train_gemma4-31b-it_text_only_vision_off.py --max-seq-length 1024 --lora-r 32 --lora-alpha 64
"""

# ============================================================
# ВАЖНО: env vars должны стоять ДО импортов torch / unsloth
# ============================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("UNSLOTH_OFFLOAD_GRADIENTS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")

# Stability mode: disable Unsloth/PyTorch auto compilation.
# This is slower, but avoids TorchDynamo recompilation crashes on variable shapes.
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

# Hard-disable TorchDynamo/torch.compile in this process.
# Some Unsloth + Torch 2.12 + Gemma4 combinations can hit recompile limits.
try:
    import torch._dynamo
    torch._dynamo.disable()
    torch._dynamo.config.suppress_errors = True
except Exception as _e:
    print(f"[WARN] Could not disable torch._dynamo cleanly: {_e}")

# Safety belt: make accidental torch.compile calls return the original function/module.
def _eva_no_compile(fn=None, *args, **kwargs):
    if fn is None:
        def decorator(inner):
            return inner
        return decorator
    return fn

try:
    torch.compile = _eva_no_compile
except Exception as _e:
    print(f"[WARN] Could not monkey-patch torch.compile: {_e}")

from datasets import Dataset

try:
    # Gemma-4 can be multimodal and contain a vision_tower.
    # Use FastVisionModel so LoRA can explicitly skip vision layers.
    from unsloth import FastVisionModel
except Exception as e:
    raise RuntimeError("Cannot import FastVisionModel from unsloth. Check your Unsloth install.") from e

try:
    from unsloth import is_bfloat16_supported
except Exception:
    def is_bfloat16_supported() -> bool:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

from unsloth.chat_templates import train_on_responses_only

try:
    from unsloth.chat_templates import get_chat_template
except Exception:
    get_chat_template = None

from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


# ============================================================
# Parameters, обычно меняешь только этот блок
# ============================================================
MODEL_NAME = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DATASET_PATH = "/home/path/you_dataset.jsonl"
OUTPUT_DIR = "/home/path/gemma4_output"

# SAFE DEFAULTS. 
MAX_SEQ_LENGTH = 2048
LORA_R = 32
LORA_ALPHA = 64

NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 32
LEARNING_RATE = 1e-4
WARMUP_STEPS = 7
SAVE_STEPS = 30
SAVE_TOTAL_LIMIT = 3
SEED = 3407

# Если пример длиннее контекста, лучше выкинуть, чем получить OOM или обрезанный assistant.
DROP_LONG_EXAMPLES = True

# Если хочешь маленький smoke-test:
# MAX_STEPS = 10
MAX_STEPS = -1

# Gemma 4 / Unsloth chat template markers:
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--data", default=DATASET_PATH)
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--epochs", type=float, default=NUM_TRAIN_EPOCHS)
    p.add_argument("--batch", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--no-drop-long", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-confirm", action="store_true")
    return p.parse_args()


def cuda_report(title: str) -> None:
    print(f"\n=== CUDA: {title} ===")
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Free:      {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
    print(f"Allocated: {allocated / 1024**3:.2f} GB")
    print(f"Reserved:  {reserved / 1024**3:.2f} GB")


def normalize_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Accepts:
      {"messages": [{"role": "user"/"assistant"/"system", "content": "..."}]}
      {"conversations": [{"from": "human"/"gpt"/"system", "value": "..."}]}
    Returns regular chat roles: system / user / assistant
    """
    if "messages" in example:
        raw = example["messages"]
        out = []
        for m in raw:
            role = m.get("role")
            content = m.get("content", "")
            if role == "model":
                role = "assistant"
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unknown role in messages: {role}")
            out.append({"role": role, "content": str(content)})
        return out

    if "conversations" in example:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "model": "assistant",
            "system": "system",
            "developer": "system",
        }
        out = []
        for m in example["conversations"]:
            role = role_map.get(m.get("from"))
            if role is None:
                raise ValueError(f"Unknown role in conversations: {m.get('from')}")
            out.append({"role": role, "content": str(m.get("value", ""))})
        return out

    raise ValueError(f"Unknown example format. Keys: {list(example.keys())}")


def has_assistant(messages: List[Dict[str, str]]) -> bool:
    return any(m["role"] == "assistant" and m.get("content", "").strip() for m in messages)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                bad += 1
                print(f"[WARN] Bad JSON at line {n}: {e}")
    if bad:
        print(f"[WARN] Skipped bad JSON lines: {bad}")
    return rows


def prepare_dataset(raw_data: List[Dict[str, Any]], tokenizer, max_seq_length: int, drop_long: bool) -> Dataset:
    good = []
    dropped_no_assistant = 0
    dropped_bad = 0

    for i, ex in enumerate(raw_data):
        try:
            messages = normalize_messages(ex)
            if not has_assistant(messages):
                dropped_no_assistant += 1
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Guard: Gemma-4 response marker must appear, otherwise response-only masking will be empty.
            if RESPONSE_PART not in text:
                dropped_bad += 1
                if dropped_bad <= 3:
                    print(f"[WARN] Example {i} has no response marker {RESPONSE_PART!r}. First 300 chars:")
                    print(text[:300].replace("\n", "\\n"))
                continue

            good.append({"text": text})
        except Exception as e:
            dropped_bad += 1
            if dropped_bad <= 5:
                print(f"[WARN] Dropping example {i}: {e}")

    if not good:
        raise RuntimeError("No usable examples after formatting. Check dataset roles and chat template.")

    ds = Dataset.from_list(good)

    # Gemma-4 Unsloth may return a processor-like object.
    # For length scan we want the inner TEXT tokenizer, no padding, no images/videos.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def add_len(batch):
        encoded = text_tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        ids = encoded["input_ids"]

        # Batched call: list[list[int]]
        if ids and isinstance(ids[0], list):
            lengths = [len(x) for x in ids]
        # Single example fallback: list[int]
        else:
            lengths = [len(ids)]

        return {"n_tokens": lengths}

    ds = ds.map(add_len, batched=True, desc="Token length scan")

    lengths = ds["n_tokens"]
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[int(0.50 * (len(lengths_sorted) - 1))]
    p95 = lengths_sorted[int(0.95 * (len(lengths_sorted) - 1))]
    p99 = lengths_sorted[int(0.99 * (len(lengths_sorted) - 1))]
    max_len = max(lengths_sorted)

    print("\n=== Dataset ===")
    print(f"Loaded raw:              {len(raw_data)}")
    print(f"Usable formatted:        {len(ds)}")
    print(f"Dropped no assistant:    {dropped_no_assistant}")
    print(f"Dropped bad/template:    {dropped_bad}")
    print(f"Token lengths:           p50={p50}, p95={p95}, p99={p99}, max={max_len}")
    print(f"Max seq length:          {max_seq_length}")

    if drop_long:
        before = len(ds)
        ds = ds.filter(lambda x: x["n_tokens"] <= max_seq_length, desc="Drop too-long examples")
        after = len(ds)
        print(f"Dropped too-long:        {before - after}")
        if after == 0:
            raise RuntimeError("All examples are longer than max_seq_length. Increase context or clean dataset.")
    else:
        print("[WARN] Not dropping long examples. SFTTrainer will truncate; this can cut assistant answers.")

    ds = ds.remove_columns(["n_tokens"])

    print("\n=== First formatted example preview ===")
    print(ds[0]["text"][:1000].replace("\n", "\\n"))
    return ds


def sanity_check_labels(trainer, n: int = 8) -> None:
    """
    After train_on_responses_only, Unsloth normally adds labels to the dataset.
    If labels are all -100, loss can be 0 / tiny and gradients near zero.
    """
    print("\n=== Response-only mask sanity check ===")
    ds = trainer.train_dataset
    checked = min(n, len(ds))
    active_counts = []
    total_counts = []

    for i in range(checked):
        row = ds[i]
        if "labels" not in row:
            print("Dataset row has no 'labels' field yet. This Unsloth/TRL version may build labels in collator.")
            print("Continuing, but watch loss and grad_norm in first steps.")
            return
        labels = row["labels"]
        active = sum(1 for x in labels if x != -100)
        total = len(labels)
        active_counts.append(active)
        total_counts.append(total)
        print(f"example {i}: active labels = {active}/{total} ({100 * active / max(total, 1):.2f}%)")

    if active_counts and max(active_counts) == 0:
        raise RuntimeError(
            "All checked examples have 0 active labels. "
            "train_on_responses_only mask is wrong for this chat template."
        )

    if active_counts:
        mean_active = sum(active_counts) / len(active_counts)
        mean_total = sum(total_counts) / len(total_counts)
        print(f"mean active labels: {mean_active:.1f}/{mean_total:.1f} ({100 * mean_active / max(mean_total, 1):.2f}%)")
        if mean_active < 8:
            print("[WARN] Very few assistant tokens are trained. Loss/grad may look almost zero.")


def main():
    args = parse_args()

    print("=== Gemma-4-31B SFT OOM-safe ===")
    print("Compilation disabled: UNSLOTH_COMPILE_DISABLE=1, TORCH_COMPILE_DISABLE=1")
    print(f"Model:          {args.model}")
    print(f"Dataset:        {args.data}")
    print(f"Output:         {args.out}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA r/alpha:   {args.lora_r}/{args.lora_alpha}")
    print(f"Batch/accum:    {args.batch}/{args.grad_accum}")
    print(f"LR:             {args.lr}")
    print(f"Drop long:      {not args.no_drop_long}")

    cuda_report("before load")

    bf16_ok = is_bfloat16_supported()
    print(f"\nbf16 supported: {bf16_ok}")

    # ============================================================
    # Load model
    # ============================================================
    print("\n=== Loading model ===")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
    )

    # Force Gemma-4 template if available. If the tokenizer already has it, this is harmless.
    if get_chat_template is not None:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
            print("Using Unsloth chat template: gemma-4")
        except Exception as e:
            print(f"[WARN] Could not force gemma-4 chat template: {e}")
            print("Using tokenizer's own chat_template.")

    if hasattr(model, "config"):
        model.config.use_cache = False

    cuda_report("after load")

    # ============================================================
    # Attach LoRA
    # ============================================================
    print("\n=== Attaching LoRA ===")
    model = FastVisionModel.get_peft_model(
        model,

        # CRITICAL: text-only SFT. Do not train vision tower.
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,

        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,

        # Let Unsloth select the proper linear modules inside the enabled parts only.
        target_modules="all-linear",
    )

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    cuda_report("after LoRA")

    # ============================================================
    # Dataset
    # ============================================================
    raw_data = load_jsonl(args.data)
    dataset = prepare_dataset(
        raw_data=raw_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        drop_long=not args.no_drop_long,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_report("before trainer")

    # ============================================================
    # Trainer
    # ============================================================
    print("\n=== Collator ===")
    print("Using DataCollatorForSeq2Seq pad_to_multiple_of=512 to reduce TorchDynamo recompiles.")
    training_args = SFTConfig(
        output_dir=args.out,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # adamw_8bit is the standard Unsloth choice.
        # If you still get optimizer memory spikes, try: optim="paged_adamw_8bit"
        optim="adamw_8bit",

        seed=SEED,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,

        bf16=bf16_ok,
        fp16=not bf16_ok,

        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=512,
            label_pad_token_id=-100,
        ),
        args=training_args,
    )

    # CRITICAL: Train only on model/assistant responses.
    # Gemma-4 template renders assistant role as <|turn>model\n
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    sanity_check_labels(trainer)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_report("before train")

    if not args.no_confirm:
        input("\nPress Enter to start training, Ctrl+C to abort...")

    print("\n=== Training ===")
    if args.resume:
        stats = trainer.train(resume_from_checkpoint=True)
    else:
        stats = trainer.train()

    print(f"\nTraining complete. Final training_loss: {stats.training_loss:.6f}")

    final_dir = Path(args.out) / "final_lora"
    print(f"\n=== Saving LoRA adapter to {final_dir} ===")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done ❤️")


if __name__ == "__main__":
    try:
        main()
    except TypeError as e:
        print("\n[TYPE ERROR]")
        print(str(e))
        print("\nLikely version mismatch between unsloth / trl / transformers.")
        print("Try upgrading cleanly:")
        print("  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo trl transformers accelerate bitsandbytes")
        raise
