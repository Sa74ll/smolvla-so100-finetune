## SmolVLA Colour Aug Challenge

Challenge goal: train **SmolVLA** on `lerobot/svla_so101_pickplace` with colour augmentation in **training** and show it still works when **validation** uses a different colour setup.


### Quick Start
- **[📔 Run Train script in Colab](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/01_train_smolvla.ipynb)** - Interactive notebook
- **[📔 Run Eval script in Colab](https://colab.research.google.com/github/Sa74ll/ELM_challenge/blob/main/02_eval_offline.ipynb)** - Interactive notebook

---
## Files

- `01_train.ipynb` – fine-tunes `lerobot/smolvla_base` 
- `02_eval_offline.ipynb` 

---

**Key Result**: Achieved **60% average per-joint success** (within 5% tolerance) under colour distribution shift.

---

## Architecture & Approach

### Model: SmolVLA
- Pre-trained vision-language-action policy from `lerobot/smolvla_base`
- Autoregressive action prediction with **chunk size = 50**
- Requires temporal alignment of observations and actions

### Dataset: SO-100 Pick-Place
- **Source**: `lerobot/svla_so101_pickplace` (50 episodes)
- **Split**: Episodes 0-39 (train) | Episodes 40-49 (val)
- **FPS**: 30 Hz
- **Cameras**: `up` and `side` views
- **Action space**: 6-DoF continuous control

### Training Strategy
- **Step-based training**: 25,000 steps (not epoch-based) to resume
- **Colour augmentation**: Different distributions for train vs. val
- **Validation**: Every 1,000 steps
- **Checkpoints**: Every 1,000 steps to resume whenever the Colab runtime crashes

Key point: `video_backend="pyav"` was used because the LeRobot issues mention torchvision video backend deprecations the warning shows up, but pyav worked reliably in Colab.

---

##  Implementation Details

### 1. Alignment with the policy challenge

**Problem**: Initial runs crashed with:
```
RuntimeError: The size of tensor a (227) must match the size of tensor b (178)
```

**Root Cause**: Model expects 50 action timesteps per forward pass, but dataset only provided 1.

**Solution**: Constructed proper `delta_timestamps` aligned with model's `chunk_size` and dataset FPS:
```python
fps = 30  # from dataset metadata
action_horizon = 50  # from policy.config.chunk_size

delta_timestamps = {                   # from robot learning tutorial
    "observation.state": [0.0],
    "observation.images.up": [0.0],
    "observation.images.side": [0.0],
    "action": [i / fps for i in range(action_horizon)]  # [0.0, 0.033, 0.066, ...]
}
```
This ensures the dataset fetches 50 future action steps at proper temporal intervals.

---

### 2. Colour Augmentation for Robustness

**Train Configuration** (normal distribution):
```python
train_transforms = ImageTransformsConfig    # from lerobot.datasets.transforms
(                  
    enable=True,
    max_num_transforms=2,
    random_order=True,
    tfs={
        "brightness": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"brightness": (0.8, 1.2)}
        ),
        "contrast": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"contrast": (0.8, 1.2)}
        ),
        "saturation": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"saturation": (0.5, 1.5)}
        ),
        "hue": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"hue": (-0.05, 0.05)}
        ),
    },
)
```
 
**Val Configuration** (shifted distribution - darker, higher contrast):
```python
val_transforms = ImageTransformsConfig    # from lerobot.datasets.transforms
( 
    enable=True,
    max_num_transforms=2,
    random_order=True,
      tfs={
        "brightness": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"brightness": (0.7, 1.0)}
        ),
        "contrast": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"contrast": (1.0, 1.3)}
        ),
        "saturation": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"saturation": (0.5, 1.2)}
        ),
        "hue": ImageTransformConfig(
            weight=1.0, type="ColorJitter", kwargs={"hue": (-0.08, 0.06)}
        ),
    }
)
```

This tests the model's ability to generalise under lighting/colour variations.

---

### 3. Episode-Based Splitting

**Why it matters**: Prevents temporal data leakage between train and val sets.
```python
episode_idx = np.array(base_ds.hf_dataset["episode_index"])
train_indices = [i for i, ep in enumerate(episode_idx) if ep < 40]
val_indices = [i for i, ep in enumerate(episode_idx) if ep >= 40]

# Final counts
# Train: 9,180 samples | Val: 2,759 samples
```

---

### 4. Camera Key Remapping

**Problem**: Dataset uses `observation.images.up` / `observation.images.side`, but SmolVLA expects `camera1` / `camera2`.

**Solution**:
```python
def fix_keys(batch):
    if "observation.images.up" in batch:
        batch["observation.images.camera1"] = batch.pop("observation.images.up")
    if "observation.images.side" in batch:
        batch["observation.images.camera2"] = batch.pop("observation.images.side")
    return batch
```

---

## Evaluation Methodology

### Challenge: Action Normalisation

**Problem #1**: Initial eval showed MAE in the thousands and success rate was 0%.

**Root Cause**: Model outputs normalised actions, but ground truth is in original units.

**Problem #2**: `policy.forward()` returns loss dict, not actions.

**Solution**: Use policy.predict_action_chunk() + proper denormalization:
```python
1-
pred_seq = policy.predict_action_chunk(batch)  # (B, 50, action_dim)
pred_action = pred_seq[:, 0, :]  # First action in chunk

2. Denormalise predictions
meta = LeRobotDatasetMetadata(DATASET_REPO)
action_mean = torch.tensor(meta.stats["action"]["mean"])
action_std = torch.tensor(meta.stats["action"]["std"])

pred_unnorm = pred_action * action_std + action_mean
```

### Final Results

**Per-Joint Success @ 5% Tolerance**:
```
joint 0: 45.81%
joint 1: 47.88%
joint 2: 70.06%
joint 3: 77.46%
joint 4: 60.86%
joint 5: 63.43%

Average per-joint success (5%): 60.92%

```
---
## References

- Course: https://huggingface.co/spaces/lerobot/robot-learning-tutorial
- SmolVLA: https://huggingface.co/blog/smolvla
- Dataset: https://huggingface.co/datasets/lerobot/svla_so101_pickplace
- LeRobot: https://github.com/huggingface/lerobot
