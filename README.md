# where_does_in-context-learning_happen_in_LLMs

This is the Official Repository for ["Where Does In-context Learning Happen in LLMs, NeurIPS 2024."](https://openreview.net/pdf?id=LLuSjg59an)

#### Abstract:

Self-supervised large language models have demonstrated the ability to perform various tasks via in-context learning, but little is known about where the model locates the task with respect to prompt instructions and demonstration examples. In this work, we attempt to characterize the region where large language models transition from recognizing the task to performing the task. Through a series of layer-wise context-masking experiments on GPTNEO2.7B, BLOOM3B, and STARCODER2-7B, LLAMA3.1-8B, LLAMA3.1-8B-INSTRUCT, on Machine Translation and Code generation, we demonstrate evidence of a "task recognition" point where the task is encoded into the input representations and attention to context is no longer necessary. Taking advantage of this redundancy results in 45% computational savings when prompting with 5 examples, and task recognition achieved at layer 14 / 32 using an example with Machine Translation. Our findings also have implication for resource and parameter efficient fine-tuning; we observe a correspondence between fine-tuning performance of individual LoRA layers and the task recognition layers.

### Data Preparation

#### Machine Translation 
> `bash bin/prepare-flores.sh`

> `bash bin/prepare-codegen.sh`

### Main Attention Context-Masking Experiments

`bin/batch..` contain for-loops which call `bin/submit...sh` over experimental conditions.

> `bash bin/batch_submit_baseline.sh`

> `bash bin/batch_submit_mask_exp.sh`

Visualising Results and Generating Figures: [ipynb](https://github.com/suzyahyah/where_does_in-context-learning_happen_in_LLMs/blob/main/ipy_nbs/display_results_context_mask_from.ipynb)


### Main Attention Context-Masking Experiments

Lora Experiments 

> `bash bin/lora_train.sh`

Visualising Results and Generating Figures: [ipynb](https://github.com/suzyahyah/where_does_in-context-learning_happen_in_LLMs/blob/main/ipy_nbs/display_lora.ipynb)

### Config Files

There are config files corresponding to each of the model masking ablations:


* `configs/model/masks/mask_context_from_{F,T}{F,T}{F,T}.yaml`

- Mask Instructions = {False, True}
- Mask Examples = {False, True}
- Mask Query = {False, True}


For instance, `configs/model/masks/mask_context_from_TTF.yaml`: mask instructions, mask examples, do not mask query.

* `configs/model/default.yaml`: used when running baselines, i.e., ceiling performance.

and also instruction variants
* `configs/format/instr_code_gen.yaml`: instructions for code gen task.
* `configs/format/instr_machine_translation.yaml`: instructions for machine translation task.
* `configs/format/instr_none_QA.yaml`: no instructions.
