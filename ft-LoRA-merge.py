from datasets import load_dataset
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
import mlflow

base_model = "bloom-1b1"
base_model_name = "bloom-1b1"
merged_model = "merged_bloom-1b1"
#base_model = "falcon-7b"
#base_model_name = "falcon-7b"
#merged_model = "merged_falcon-7b"
training_output = "train_dspeed_2w" # stores the checkpoints
dataset_name = "text-to-sql_dataset"
split = "train[:10%]" # train the first 10% of the dataset
device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
trainlogs = "trainlogs"

# BitsAndBytesConfig config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

#bnb_config = BitsAndBytesConfig(
#    load_in_8bit=True,
#)

peft_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
)

def prompt_instruction_format(sample):
  return f"""Context:
    {sample['instruction']}

    Result:
    {sample['output']}
    """

dataset = load_dataset(dataset_name, split=split)

#optimizer = torch.optim.Adam(model.parameters())
#data = torch.utils.data.DataLoader(dataset, shuffle=True)

#base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, use_cache = False, device_map=device_map)
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map) #use_cache=true involves more memory

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

trainingArgs = TrainingArguments(
    output_dir=training_output,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    #auto_find_batch_size=True,
    #gradient_checkpointing=True, # When enabled, memory can be freed at the cost of small decrease in the training speed due to recomputing parts of the graph during back-propagation. DDP has issue when this is set to True.
    #gradient_accumulation_steps=1, # Smaller values lead to more frequent gradient updates, speed up convergence. Higher values prevents OOM.
    #optim="paged_adamw_32bit",
    #optim="paged_adamw_8bit",
    #optim="adamw_bnb_8bit",
    logging_steps=5, #print status every x steps
    save_strategy="epoch",
    logging_dir=trainlogs,
    #learning_rate=2e-4,
    #fp16=True, # Use mixed precision instead of 32-bit. Default=False
    #bf16=True, # Use mixed precision instead of 32-bit. Reduce memory by a fraction. Requires Ampere or higher NVIDIA. Default=False
    report_to=["tensorboard"],
    disable_tqdm=True
)


trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=trainingArgs,
)


print("Start Fine-Tuning")
#mlflow.set_experiment("Deepspeed bloom-1b1")
trainer.train()
print("Training Done")

trainer.save_model() # adapter models
print("Model Saved")

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    trainingArgs.output_dir,
    return_dict=True,
    device_map=device_map
)

# Merge LoRA adapter with the base model and save the merged model

lora_merged_model = trained_model.merge_and_unload()
#trained_model.cpu().save_pretrained(merged_model)
lora_merged_model.save_pretrained(merged_model)
tokenizer.save_pretrained(merged_model)