# SupervisedVsLLM-EfficacyEval
This is the data and code for the paper: Evaluating the Efficacy of Supervised Learning vs. Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media.
## Directory Organization

- data/
    - cognitive distortion/ : Data containing cognitive distortions has been stored.There are data files of various formats in this folder for subsequent use by different models.
    - suicide/ : Data containing suicide has been stored.There are data files of various formats in this folder for subsequent use by different models.
- bert-base-chinese/ : BERT's pre-trained files.
- LLM/ : code for LLM test.
  - ZeroShot/ : The code for zero-shot prompting.This process encompasses various strategies, including direct task requests (acting as the basic), role-definition, scene-definition, and hybrid approaches.
  - FewShot/ : The code for few-shot prompting.Few-shot prompting integrates prior knowledge, a batch of n training instances, and their combination, to pose queries in conjunction with one of the four strategies in zero-shot testing.
  - Fine-tuning/ : The code for LLM-fine-tunning.
- supervised_learning/ : Code for supervised learning. Includes two major supervised learning models: BERT and LSAN.

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
