# Pep-Instruction

This repository contains the implementation for fine-tuning LLMs specifically on **peptide sequences**. It includes source code for training, peptide datasets, and evaluation codes for various downstream tasks.

## Repository Structure

The repository is organized as follows:

- **Function-Prediction/**
  Contains test datasets and evaluation codes for peptide function prediction .

- **Design/**
  Contains test datasets and evaluation codes for peptide sequence design.

- **Optimization/**
  Contains test datasets and evaluation codes for peptide sequence optimization tasks.

- **Property-Prediction/**
  Contains test datasets and evaluation codes for peptide property prediction.

- **zeroshot/**
  Contains specific datasets and codes for **Zero-Shot evaluation** on the properties covered in the Property Prediction task.
  
##usage
### Training
The `finetune.py` code located in the `code` directory is used for fine-tuning the model on peptide datasets.

### Models

The fine-tuned model weights are available on Hugging Face: [Pep-instruction](https://huggingface.co/Codelife176/Pep-instruction)
