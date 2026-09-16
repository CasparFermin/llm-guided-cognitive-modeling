# llm-guided-cognitive-modeling

This project consists of five stages:

1. Loading and preprocessing the data, done with loadAndSplitData.py
2. Downloading an LLM and fine-tuning LoRA's and merging an appropriate one with the original LLM (the analysis.ipynb contains a ).
3. Performing LLM inference through simple next token prediction to generate logits for the test-set.
4. Running OpenEvolve twice, both with the base LLM and fine-tuned LLM to generate cognitive models based on the training-set.
5. Parameter training and testing the generated cognitive models which yields logits for the test-set.
6. Parameter trianing and testing the Rescorla-Wagner model which yields logits for the test-set.
7. Performing the analysis.

Steps 2, 3, and 4 were performed on a Snellius node with H100's, the paths have to be changed accordingly to succesfully run these scripts.

The other steps can be readily performed using the provided files, although the analysis does assume you have the data-set containing the predictions on the test-set of all 5 models.

It is also possible to [download ]([huggingface.co/datasets/cObliUvA/llm-guided-modelling-results/tree/main](https://huggingface.co/datasets/cObliUvA/llm-guided-modelling-results/tree/main))the test-set logit files and reproduce the exact results of the paper.

Moreover, the fine-tuning can also be skipped, as the LoRA's can be [downloaded ](https://huggingface.co/cObliUvA/Qwen3-Coder-Next-LoRA/tree/main)to be merged with [Qwen3-Coder-Next 80B A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next).


*Note*: Four participants were removed from the Drifting Bandits test-set due to not answer all options, they had ID's: 712, 35, 474, 187. This is no longer neccesary if the project is ran from scratch as this bug has been removed.
