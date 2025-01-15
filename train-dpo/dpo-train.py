base_model = "Qwen/Qwen2.5-7B-Instruct"
sft_adapter = "../train-sft/out_trainer"

from trl import DPOConfig
from trl.trainer import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import AutoTokenizer, Qwen2ForCausalLM
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import LoraConfig, get_peft_model
import torch

model = AutoModelForCausalLM.from_pretrained(base_model)
model.enable_input_require_grads()
# model.load_adapter("/data/wsw/new_train copy/out_trainer", "sft")
from peft.peft_model import PeftModelForCausalLM
model = PeftModelForCausalLM.from_pretrained(
    model,
    sft_adapter, 
    is_trainable=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
training_args = DPOConfig(
    torch_empty_cache_steps=1000,
    output_dir="dpo_res", 
    logging_steps=10, 
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    bf16=True,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    beta=0.08,
    num_train_epochs=1,
    save_steps=10000,
)
j = json.load(open("/data/wsw/eval_lora_new/for_doing_dpo/dpo_data.json"))
rr = []
for each in j:
    if len(each['prompt']) > 4096 or len(each['rejected']) > 4096 or len(each['chosen']) > 4096:
        continue
    else:
        rr.append(each)
ds = Dataset.from_list(rr)
ds = ds.shuffle(seed=42)
trainer = DPOTrainer(
    model=model,
    train_dataset=ds,
    args=training_args,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./out_dpo")
