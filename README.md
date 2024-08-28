
# Fine-Tune LLaMA Story Teller

This repository contains a script for fine-tuning the LLaMA model using a custom dataset and specialized configurations. It leverages advanced techniques like quantization and low-rank adaptation (LoRA) to optimize training and performance.

## Overview

The script performs the following tasks:
1. **Load Dataset**: Downloads and prepares the dataset.
2. **Initialize Tokenizer and Model**: Configures the tokenizer and sets up the LLaMA model with 4-bit quantization.
3. **Configure LoRA**: Sets up low-rank adaptation for efficient training.
4. **Set Training Arguments**: Configures the training parameters.
5. **Train Model**: Fine-tunes the model on the dataset and uploads it to the model hub.

## Dependencies

- `torch`
- `datasets`
- `peft`
- `transformers`
- `trl`

Ensure you have these dependencies installed. You can use `pip` to install them:

```bash
pip install torch datasets peft transformers trl
```

Usage

To fine-tune the LLaMA model, run the script:

```bash
python finetune_llama_story_teller.py
```

## Script Details
1. **Loading Data**: The load_data function fetches the dataset specified by dataset_name and split_name.
2. **Tokenization**: The initialize_tokenizer function sets up the tokenizer and configures the padding token.
3. **Model Initialization**: The initialize_model function initializes the LLaMA model with 4-bit quantization for efficient computation.
4. **LoRA Configuration**: The configure_lora function sets up LoRA for low-rank adaptation to enhance model performance.
5. **Training Arguments**: The configure_training_arguments function specifies parameters such as batch size, learning rate, and number of epochs.
6. **Training**: The train_model function uses SFTTrainer to fine-tune the model and push the results to the model hub.

## Configuration

1. **Dataset**: The script uses the dataset "2173ars/finetuning_story". Ensure that you have access to this dataset or modify the dataset_name variable accordingly.
2. **Model**: The script uses "meta-llama/Llama-2-7b-hf". You can replace this with a different model if needed.
3. **Training Parameters**: Adjust the training parameters in configure_training_arguments based on your requirements.

## Contributing

Feel free to open issues or submit pull requests if you have improvements or suggestions.

