# Being-H

Being-H is BeingBeyond's family of human-centric embodied foundation models.
Within this repository, **Being-H0.5** is our flagship **VLA** model and **Being-H0.7** is our flagship **WAM** model.

## Model Family

| Project&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Positioning&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Summary | Links&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---------|-------------|---------|-------|
| [Being-H0.7](Being-H07/) | Flagship WAM | A latent world-action model from egocentric videos with future-aware latent reasoning. | [Blog](https://research.beingbeyond.com/being-h07) / [Paper](https://research.beingbeyond.com/projects/being-h07/being-h07.pdf) |
| [Being-H0.5](Being-H05/) | Flagship VLA | A human-centric VLA model for cross-embodiment generalization with a unified action space. | [Blog](https://research.beingbeyond.com/being-h05) / [Paper](https://arxiv.org/pdf/2601.12993) / [Models](https://huggingface.co/collections/BeingBeyond/being-h05) |
| [Being-H0](https://github.com/BeingBeyond/Being-H/tree/being-h0) | Previous VLA | The first Being-H release for human-video VLA pretraining. | [Blog](https://research.beingbeyond.com/being-h0) / [Paper](https://arxiv.org/pdf/2507.15597) / [Models](https://huggingface.co/collections/BeingBeyond/being-h0) |

## News

- **[2026-04-14]**: We publish **Being-H0.7**, our flagship WAM model. See the [blog](https://research.beingbeyond.com/being-h07) and [paper](https://research.beingbeyond.com/projects/being-h07/being-h07.pdf).
- **[2026-03-20]**: We release the [UniHand_Preview](https://huggingface.co/datasets/BeingBeyond/UniHand_Preview) dataset, a subset of the Being-H0.5 pre-training mixture.
- **[2026-01-24]**: We update the H0.5 training, inference, and data preparation docs, and open-source post-training data for PND Adam-U through our [Hugging Face dataset collection](https://huggingface.co/collections/BeingBeyond/pnd-adam-u-data).
- **[2026-01-20]**: We publish **Being-H0.5**, our flagship VLA model for cross-embodiment generalization.
- **[2025-08-02]**: We release the **Being-H0** codebase and pretrained models through the [BeingBeyond Hugging Face collections](https://huggingface.co/collections/BeingBeyond/being-h0).
- **[2025-07-21]**: We publish **Being-H0**, our first human-video VLA release. Read the [paper](https://arxiv.org/pdf/2507.15597).


## Projects Based on Being-H

We are seeing a growing set of excellent projects built on top of the Being-H family:

- Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting. [arXiv 26'03](https://arxiv.org/abs/2603.16542) | [website](https://research.beingbeyond.com/ptr) | [GitHub](https://github.com/BeingBeyond/PTR)
- DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation. [arXiv 26'03](https://arxiv.org/abs/2603.09121) | [website](https://chenzhongxi-sjtu.github.io/dexhil/)
- Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild. [arXiv 26'02](https://arxiv.org/abs/2602.21736) | [website](https://research.beingbeyond.com/jala) | [GitHub](https://github.com/BeingBeyond/JALA)
- Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization. [arXiv 26'02](https://arxiv.org/pdf/2602.09722) | [website](https://research.beingbeyond.com/rethink_vla) | [GitHub](https://github.com/BeingBeyond/Rethink_VLA)

Feel free to open a pull request if you want to share work built on Being-H.

## Citation

If you find the Being-H family useful, please consider citing the relevant release:

**Being-H0.7**

```bibtex
@misc{beingbeyond2026beingh07,
  title={Being-H0.7: A Latent World-Action Model from Egocentric Videos},
  author={BeingBeyond Team},
  year={2026},
  howpublished={\url{https://research.beingbeyond.com/being-h07}}
}
```

**Being-H0.5**

```bibtex
@article{beingbeyond2026beingh05,
  title={Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization},
  author={Luo, Hao and Wang, Ye and Zhang, Wanpeng and Zheng, Sipeng and Xi, Ziheng and Xu, Chaoyi and Xu, Haiweng and Yuan, Haoqi and Zhang, Chi and Wang, Yiqing and Feng, Yicheng and Lu, Zongqing},
  journal={arXiv preprint arXiv:2601.12993},
  year={2026}
}
```

**Being-H0**

```bibtex
@article{beingbeyond2025beingh0,
  title={Being-h0: vision-language-action pretraining from large-scale human videos},
  author={Luo, Hao and Feng, Yicheng and Zhang, Wanpeng and Zheng, Sipeng and Wang, Ye and Yuan, Haoqi and Liu, Jiazheng and Xu, Chaoyi and Jin, Qin and Lu, Zongqing},
  journal={arXiv preprint arXiv:2507.15597},
  year={2025}
}
```

## License

This repository is released under Apache-2.0. See [LICENSE](LICENSE).
