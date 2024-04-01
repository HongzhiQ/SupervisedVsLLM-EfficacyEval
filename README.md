# Evaluating the Efficacy of Supervised Learning vs. Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media

This is the data and code for the paper: Evaluating the Efficacy of Supervised Learning vs. Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media.

* Paper link: https://arxiv.org/abs/2309.03564

## Introduction

We introduce two novel annotated datasets from Chinese social media, focused on cognitive distortions and suicidal risk classification. We propose a comprehensive benchmark using both traditional supervised learning and large language models, especially from the GPT series, to evaluate performance on these datasets. To assess the capabilities of the large language models, we employed three strategies: zero-shot, few-shot, and fine-tuning.

The specific code directory structure is as follows:

- `data/`:
  - `cognitive distortion/`:Data containing cognitive distortions has been stored. There are data files of various formats in this folder for subsequent use by different models.
  - `suicide/`: Data containing suicide has been stored. There are data files of various formats in this folder for subsequent use by different models.
- `bert-base-chinese/`: BERT's pre-trained files. Due to file size limitations, you can download it from the website: https://huggingface.co/bert-base-chinese# .
- `LLM/`:code for LLM test.
  - `ZeroShot/`:The code for zero-shot prompting. This process encompasses various strategies, including direct task requests (acting as the basic), role-definition, scene-definition, and hybrid approaches.
  - `FewShot/`: The code for few-shot prompting. Few-shot prompting integrates prior knowledge, a batch of n training instances, and their combination, to pose queries in conjunction with one of the four strategies in zero-shot testing.
  - `Fine-tuning/`: The code for LLM-fine-tunning.
- `supervised_learning/`: Code for supervised learning. Includes two major supervised learning models: BERT and LSAN.

## Dataset Update

In our subsequent research,We have expanded the cognitive distortion dataset and meticulously annotated it. If you are interested in this dataset, please contact guanghui.fu@inria.fr to obtain it.
## Citation

If this repository helps you, please cite this paper:

```bibtex
@misc{qi2023evaluating,
      title={Evaluating the Efficacy of Supervised Learning vs Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media}, 
      author={Hongzhi Qi and Qing Zhao and Changwei Song and Wei Zhai and Dan Luo and Shuo Liu and Yi Jing Yu and Fan Wang and Huijing Zou and Bing Xiang Yang and Jianqiang Li and Guanghui Fu},
      year={2023},
      eprint={2309.03564},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
