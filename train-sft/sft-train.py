from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import AutoTokenizer, Qwen2ForCausalLM
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import LoraConfig, get_peft_model
import torch

data = load_from_disk("../construction-sft/train")
data = data.shuffle(seed=42)


base_model = "Qwen/Qwen2.5-7B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(base_model)
lora = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"],
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    use_rslora=True,
)
model.enable_input_require_grads()
model = get_peft_model(model, lora)
tokenizer = AutoTokenizer.from_pretrained(base_model)

args = TrainingArguments(
    report_to="wandb",
    do_train=True,
    do_eval=False,
    do_predict=False,
    per_device_train_batch_size=1,
    learning_rate=1e-2,
    lr_scheduler_type="cosine",
    num_train_epochs=2,
    output_dir="./output",
    # optim="lomo",
    logging_steps=50,
    bf16=True,
    save_steps=25000,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    resume_from_checkpoint=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./out_trainer")

torch.save(model.state_dict(), "./out_state.pth")
