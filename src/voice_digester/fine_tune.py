"""Modal entrypoint: LoRA SFT of Gemma-3-4B-it on the generated dataset (D019).

    uv run modal run --detach -m voice_digester.fine_tune --config-file-name ft-gemma-3-4b-lora.yaml

Trains on data/train/{train,val}.jsonl (D018) with the SAME system prompt /
user instruction the runtime uses (pulled from the digest config named in the
training config) — no train/serve skew. Loss is masked to the assistant JSON
tokens only, mirroring the logbook's collator. Adapter lands on the volume at
/vol/checkpoints/<run_name>; export_gguf.py turns it into the deployable Q4.
"""

import json
from pathlib import Path

import modal

from .config import DigestConfig, TrainingConfig
from .modal_infra import get_app, get_image, get_volume

VOL = "/vol"

app = get_app("voice-digester-fine-tune")
image = (
    get_image()
    # pillow/torchvision: gemma-3-4b-it is multimodal; its processor imports them
    .uv_pip_install("trl>=0.12", "peft>=0.13", "datasets>=2.0", "accelerate>=0.34",
                    "pillow", "torchvision")
    .add_local_python_source("voice_digester")
    .add_local_dir("data/train", "/root/data/train")
)
volume = get_volume("voice-digester")


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@app.function(
    image=image,
    gpu="L40S",
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3 * 60 * 60,
)
def fine_tune(config: TrainingConfig, prompts: DigestConfig):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    # Regex-scope LoRA to the language model — the plain suffix list would also
    # attach (dead) adapters to the vision tower's q/k/v projections.
    target_regex = rf".*language_model.*\.({'|'.join(config.lora_target_modules)})$"
    model = get_peft_model(model, LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=target_regex, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    system = prompts.rendered_system_prompt()

    def prompt_messages(sample):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{prompts.user_instruction}\n\n{sample['transcript']}"},
        ]

    def collate(samples):
        conversations = [
            prompt_messages(s)
            + [{"role": "assistant", "content": json.dumps(s["gold"], ensure_ascii=False)}]
            for s in samples
        ]
        batch = tokenizer.apply_chat_template(
            conversations, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=False, padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        # Loss only on the assistant JSON tokens (house style — see logbook collator).
        for i, s in enumerate(samples):
            prompt_len = tokenizer.apply_chat_template(
                [prompt_messages(s)], tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100
        batch["labels"] = labels
        return batch

    tokenizer.padding_side = "right"
    train_ds = Dataset.from_list(_load_rows(Path("/root/data/train/train.jsonl")))
    val_ds = Dataset.from_list(_load_rows(Path("/root/data/train/val.jsonl")))

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        args=SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps or -1,  # -1 = use epochs; >0 = smoke test
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            seed=config.seed,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    trainer.train()

    save_dir = Path(config.output_dir) / config.run_name
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    volume.commit()
    print(f"saved adapter + tokenizer to {save_dir}")


@app.local_entrypoint()
def main(config_file_name: str = "ft-gemma-3-4b-lora.yaml", max_steps: int | None = None):
    config = TrainingConfig.from_yaml(config_file_name)
    if max_steps:  # smoke-test override
        config.max_steps = max_steps
        config.run_name = f"smoke-{config.run_name}"
    prompts = DigestConfig.from_yaml(config.digest_config_file)
    print(f"run_name: {config.run_name}")
    fine_tune.remote(config, prompts)
