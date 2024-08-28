import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def load_data(dataset_name, split_name):
    """Load the dataset from the specified name and split."""
    return load_dataset(dataset_name, split=split_name)

def initialize_tokenizer(model_name):
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
    return tokenizer

def initialize_model(model_name, bnb_config):
    """Initialize the model with 4-bit quantization."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0}  # Map model to GPU 0
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def configure_lora():
    """Configure LoRA for the model."""
    return LoraConfig(
        r=64,  # Rank of low-rank adaptation
        lora_alpha=32,  # Scaling factor for LoRA
        lora_dropout=0.05,  # Dropout rate for LoRA
        bias="none",  # No bias term in LoRA layers
        task_type="CAUSAL_LM"  # Causal Language Modeling
    )

def configure_training_arguments():
    """Set up the training arguments."""
    return TrainingArguments(
        output_dir="llama2_finetuned_chatbot",
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=20,
        fp16=True,
        push_to_hub=True
    )

def train_model(model, data, tokenizer, peft_config, training_args):
    """Initialize the trainer and train the model."""
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        training_arguments=training_args,
        tokenizer=tokenizer,
        packing=False
    )
    trainer.train()
    trainer.push_to_hub()

def finetune_llama_story_teller():
    """Main function to fine-tune the LLaMA model."""
    dataset_name = "2173ars/finetuning_story"
    model_name = "meta-llama/Llama-2-7b-hf"

    # Load dataset and initialize components
    data = load_data(dataset_name, "train")
    tokenizer = initialize_tokenizer(model_name)
    
    # Configure quantization and model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    model = initialize_model(model_name, bnb_config)
    
    # Configure LoRA and training arguments
    peft_config = configure_lora()
    training_args = configure_training_arguments()
    
    # Train and push model to hub
    train_model(model, data, tokenizer, peft_config, training_args)

if __name__ == "__main__":
    finetune_llama_story_teller()
