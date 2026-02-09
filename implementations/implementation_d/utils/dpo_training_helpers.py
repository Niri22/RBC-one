# dpo_training_helpers.py
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig


def extract_prompt_from_conversations(convs):
    """
    convs: [{"from": "human", "value": "..."}]
    Returns the human prompt string.
    """
    if isinstance(convs, list) and len(convs) > 0:
        first = convs[0]
        if isinstance(first, dict) and "value" in first:
            return first["value"]
    return str(convs)


def preprocess_dpo(example):
    prompt = extract_prompt_from_conversations(example.get("conversations"))

    out = {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

    # Keep metadata (safe when remove_unused_columns=False)
    for k in ("pair_type", "test_id", "tag", "chosen_id"):
        if k in example:
            out[k] = example[k]

    return out


def load_unsloth_model(
    model_name: str,
    max_seq_length: int,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        device_map=None,   # handled by Accelerate
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_seq_length

    model.config.use_flash_attention_2 = True
    model.config.max_position_embeddings = max_seq_length

    return model, tokenizer


def apply_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        random_state=3407,
    )


def build_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    max_seq_length,
):
    PatchDPOTrainer()

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=0.1,
        num_train_epochs=3,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,

        learning_rate=1.8e-6,
        warmup_ratio=0.25,
        lr_scheduler_type="cosine",

        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
        padding_value=tokenizer.pad_token_id,

        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        save_total_limit=3,

        dataloader_num_workers=2,

        remove_unused_columns=False,
        ddp_find_unused_parameters=False,

        report_to="none",
        resume_from_checkpoint=True,
        seed=3407,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    return trainer
