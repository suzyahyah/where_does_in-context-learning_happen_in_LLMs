# where_does_in-context-learning_happen_in_LLMs
This is the Official Repository for the NEURIPS 2024 Paper "Where Does In-context Learning Happen in LLMs"

### Data Preparation

#### Machine Translation 
> `bash bin/prepare-flores.sh`

### Main Attention Context-Masking Experiments

`bin/batch..` contain for-loops which call `bin/submit...sh` over experimental conditions.

> `bash bin/batch_submit_baseline.sh`
> `bash bin/batch_submit_mask_exp.sh`

### Config Files

There are config files corresponding to each of the model masking ablations:

* `configs/model/masks/mask_context_from_FT.yaml`: dont mask instructions, mask examples.
* `configs/model/masks/mask_context_from_TT.yaml`: mask instructions, mask examples.
* `configs/model/default.yaml`: used when running baselines, i.e., ceiling performance.

and also instruction variants
* `configs/format/instr_code_gen.yaml`: instructions for code gen task.
* `configs/format/instr_machine_translation.yaml`: instructions for machine translation task.
* `configs/format/instr_none_QA.yaml`: no instructions.
