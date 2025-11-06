## SmolVLA Color Aug Challenge

Challenge goal: train **SmolVLA** on `lerobot/svla_so101_pickplace` with color augmentation in **training** and show it still works when **validation** uses a different color setup.


### Quick Start
- **[📔 Run Train script in Colab](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/01_train_smolvla.ipynb)** - Interactive notebook
- **[📔 Run Eval script in Colab](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/02_eval_offline.ipynb)** - Interactive notebook

---

## Files

- `01_train.ipynb` – fine-tunes `lerobot/smolvla_base` 
- `02_eval_offline.ipynb` 

(You can open both in Colab.)

---

## Run in Colab

1. Open:
   - Train: **[colab link](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/01_train_smolvla.ipynb)**
   - Eval: **[colab link](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/02_eval_offline.ipynb)**
2. `Runtime → Change runtime type → GPU`
3. Run all cells.

---

## What I did

- Dataset: `lerobot/svla_so101_pickplace`
- Split by **episode**: train 0–39, val 40–49
- Train augs: wider color jitter
- Val augs: slightly darker different / narrower
- Camera keys mapped: `observation.images.up/side → observation.images.camera1/2`
- Pre/post-processors built from the policy config + dataset stats

---

## Results

- Success: **60.9%**
- GPU: Colab T4

---

## References

- Course: https://huggingface.co/spaces/lerobot/robot-learning-tutorial
- SmolVLA: https://huggingface.co/blog/smolvla
- Dataset: https://huggingface.co/datasets/lerobot/svla_so101_pickplace
