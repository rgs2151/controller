# Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control

[![Paper](https://img.shields.io/badge/arXiv-2604.19018-b31b1b.svg)](https://arxiv.org/abs/2604.19018)
[![Website](https://img.shields.io/badge/Website-Visit-blue.svg)](https://trustworthyrobotics.github.io/a_lqr_site/)

This repository contains the core implementations of **A-LQR** and **S-PID**, located in the `steer` directory.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/trustworthyrobotics/lqr-activation-steering.git
cd lqr-activation-steering

# Install dependencies
pip install -r requirements.txt
```

Make sure to update 'steer/config/config.yaml' with the desired filepaths.

---

## Jacobian Collection

Each submodule includes a `<name>_data_script.py` file that handles contrastive vector construction and Jacobian collection.

These computations are best performed offline, as memory requirements can be high. If needed, you can replace calls to `collect_jacobians` with `collect_jacobians_vram` for reduced memory usage.

```bash
# Toxicity
python -m steer.toxicity.tox_data_script --model=gemma2b  # replace with your model of choice

# Truthfulness
python -m steer.truthfulness.tqa_data_script --model=gemma2b

# Concept Steering (see script for available concepts)
python -m steer.concepts.con_data_script --model=gemma2b

# Refusal
python -m steer.refusal.ref_data_script --model=gemma2b
```

---

## Evaluation

Toxicity and Truthfulness (TQA) include dedicated evaluation scripts with auxiliary benchmarks.  
Concept Steering and Refusal use slightly different evaluation pipelines.

```bash
# Toxicity
python -m steer.tox_eval        # A-LQR
python -m steer.tox_evalPID     # S-PID

# Truthfulness
python -m steer.tqa_eval        # A-LQR
python -m steer.tqa_evalPID     # S-PID

# Concept Steering (see script for available concepts)
python -m steer.concepts.concept_pipeline   # A-LQR

# Refusal
python -m steer.refusal.ref_asr_script --model=gemma2b --steering=lqr  # A-LQR+
python -m steer.refusal.ref_asr_script --model=gemma2b --steering=pid  # PID
```

Each submodule also includes a `test_<name>.py` script for debugging and experimentation.  
Notably, `test_ref.py` provides an out-of-the-box interactive demo.

```bash
python -m steer.refusal.test_ref # A-LQR+

```

---

## Citation

If you use this project in your work, please cite:

```bibtex
@article{skifstad2026local,
  title={Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control},
  author={Skifstad, Julian and Yang, Annie Xinyue and Chou, Glen},
  journal={arXiv preprint arXiv:2604.19018},
  year={2026}
}
```

---

## Contact

For questions or collaboration:

- Name: Julian Skifstad  
- Email: jskifstad3@gatech.edu  
- GitHub: https://github.com/jaskifstad
