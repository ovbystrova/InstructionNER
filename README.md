# InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER
Unofficial implementation of [InstructionNER](https://arxiv.org/pdf/2203.03903v1.pdf).

![Screenshot](resources/overall_intro.jpg)

## Requirements
Python >=3.8

## Installation
```shell
pip install -r requirements.txt
pip install -r requirements_test.txt
```

## Training
Script for training T5 model:
```
python train.py \
--path_to_instructions 'instructions.json' \
--path_to_options 'options.json' \
--log_dir 'runs/test_run' \
--eval_every_n_batches 200 \
--pred_every_n_batches 200 \
--path_to_model_config 'config.yaml' \
--path_to_model_save 'runs/model/' \
```

Arguments:
- **--path_to_instructions** - file with instruction prompts
- **--path_to_options** - file with mapping dataset to its entities
- **--log_dir** - where to log tensorboard ([BIO-markup](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
- **--eval_every_n_batches** - do evaluation every n batches ([BIO-markup](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
- **--pred_every_n_batches** - write n sample prediction every n batches
- **--config.yaml** - path to all necessary information for model
- **--path_to_model_save** - where to save model

## Prediction Sample
```
Sentence: The protest , which attracted several thousand supporters , coincided with the 18th anniversary of Spain 's constitution .
Instruction: please extract entities and their types from the input sentence, all entity types are in options
Options: ORG, PER, LOC

Prediction (raw text): Spain is a LOC.
```

# Citation
@article{wang2022instructionner,
  title={Instructionner: A multi-task instruction-based generative framework for few-shot ner},
  author={Wang, Liwen and Li, Rumei and Yan, Yang and Yan, Yuanmeng and Wang, Sirui and Wu, Wei and Xu, Weiran},
  journal={arXiv preprint arXiv:2203.03903},
  year={2022}
}