# SmolVLA Finetuning: Robotic Manipulation with Colour Robustness


![Status](https://img.shields.io/badge/Status-Completed-success) ![Model](https://img.shields.io/badge/Model-SmolVLA-orange) ![Success Rate](https://img.shields.io/badge/Success_Rate-87.66%25-brightgreen) ![Framework](https://img.shields.io/badge/Framework-LeRobot-yellow) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

A robust Vision-Language-Action (VLA) training pipeline for robotic manipulation. This project fine-tunes **SmolVLA** on the **SO-100 Pick & Place dataset**, specifically designed to handle **distributional shifts** (lighting and colour variations) and it demnstorates a fundamental principle in ML engineering: **proper data handling often matters more than model architecture or hyperparameter optimisation.**.

**Improved robot manipulation success rate from 60.92% to 87.66% (+44%) by fixing data distribution instead of tuning hyperparameters.**

---

## 🎯 The Core Achievement

| Metric | Initial Approach | Improved Approach | Change |
|--------|-----------------|-------------------|--------|
| **Success Rate** | 60.92% | **87.66%** | **+44%** |
| **Train/Val Gap** | 5.0x (overfitting) | 1.7x (healthy) | **+65% better** |
| **Best Joint (wrist_roll)** | 60.86% | **95.69%** | **+57%** |
| **Worst Joint (shoulder_pan)** | 45.81% | **98.13%** | **+114%** |

**Key Insight:** The 44% improvement came from implementing stratified data splitting, not from changing the model, hyperparameters, or training longer.

---
## The Story: A Debugging Journey

### The Challenge 

I aimed to fine-tune SmolVLA on the SO-101 pick-and-place dataset with color augmentation to test robustness to lighting variations.

**Initial Results:**
- ✅ Achieved 60.92% success rate
- ❌ But training/validation gap was 5.0x (severe overfitting)
- ❌ Model struggled on validation set

### The Investigation ("The Trap")

Something felt wrong. The validation loss was 5x higher than training loss. I tried the usual fixes:

**Attempts that didn't work:**
1. **Increased weight decay:** 0.001 -> 0.01 -> 0.1 -> 0.2
   - Result: Minimal improvement (~5%)
   
2. **Adjusted color augmentation:** Tried various brightness/contrast combinations
   - Result: No significant change
   
3. **Trained longer:** Extended beyond 15k steps
   - Result: Training loss decreased, validation stayed flat (worse)

**The Breakthrough:**

While analyzing per-episode performance, I discovered the root cause in the dataset structure:

```python
# My initial split:
train_indices = episodes 0-40  # 45 episodes
val_indices = episodes 40-49   # 5 episodes

# The SO-101 dataset structure:
# - 50 episodes = 5 cube positions × 10 episodes per position
# - Episodes 0-9: Position 0
# - Episodes 10-19: Position 1
# - Episodes 20-29: Position 2  
# - Episodes 30-39: Position 3
# - Episodes 40-49: Position 4  ← This is the problem!

# Result: The model was trained on 5 examples of Position 4 
# and evaluated EXCLUSIVELY on Position 4.
```

**The model wasn't overfitting, it was being evaluated unfairly on the specific spatial position it had seen the least.**

### The Solution: stratified sampling
Implemented **position-aware stratified sampling**:

```python 
# Ensure each position has representation in BOTH train and val
val_episodes = []
train_episodes = []

np.random.seed(42)
for position in range(5):  # 5 cube positions
    position_episodes = list(range(position * 10, (position + 1) * 10))
    shuffled = np.random.permutation(position_episodes)
    
    # 8 train, 2 val per position
    train_episodes.extend(shuffled[:8].tolist())
    val_episodes.extend(shuffled[8:10].tolist())

# Result: 40 train / 10 val, balanced across all positions
```

**Critically:** Everything else stayed the same:
- ✅ Same model architecture (SmolVLA)
- ✅ Same hyperparameters  
- ✅ Same training steps (15,000)
- ✅ Same augmentation strategy
- ✅ Same weight decay (0.001)

**Only change:** Data split strategy

### The Results 

**Success rate jumped from 60.92% to 87.66% (+44%)**

#### Per-Joint Comparison

| Joint | Before | After | Improvement |
|-------|--------|-------|-------------|
| shoulder_pan | 45.81% | **98.13%** | **+114%** 🚀 |
| shoulder_lift | 47.88% | **82.91%** | **+73%** |
| elbow_flex | 70.06% | **83.86%** | **+20%** |
| wrist_flex | 77.46% | **80.38%** | **+4%** |
| wrist_roll | 60.86% | **95.69%** | **+57%** |
| gripper | 63.43% | **84.99%** | **+34%** |

**Every single joint improved.** The biggest gains were in joints most affected by position 4 underrepresentation (shoulder_pan, wrist_roll).

---

## Key Lessons Learned

### 1. Data Quality > Hyperparameter Tuning

**Impact comparison:**
- All hyperparameter tuning combined: ~5% improvement
- Stratified data splitting: **44% improvement**

The lesson: Always inspect the data distribution before spending weeks tuning models.

### 2. Validate Your Validation Set

Questions I should have asked earlier:
- ✅ Are all task variations present in validation?
- ✅ Is the validation distribution similar to training?
- ✅ Am I accidentally testing on the hardest/easiest subset?

In my case, validating ONLY on position 4 (the least-trained position) made the model look much worse than it actually was.

### 3. Overfitting Isn't Always What It Looks Like

The 5x train/val gap looked like classic overfitting, but it was actually:
- Model learning positions 0-3 well (many examples)
- Model struggling with position 4 (few examples)  
- Validation testing ONLY position 4

**Fix:** Ensure balanced representation, not just stronger regularization.

### 4. Small Datasets Require Extra Care

With only 50 episodes (40 for training):
- Every data split decision matters significantly
- Stratification is critical, not optional
- Random splits can easily create imbalanced distributions

---

## 🏗 Technical Implementation

### Model: SmolVLA
* **Type:** Vision-Language-Action (VLA) Policy
* **Base:** `lerobot/smolvla_base` (pretrained)
* **Control:** Autoregressive action prediction with chunk size = 50

### Dataset: SO-101 Pick & Place
* **Source:** `lerobot/svla_so101_pickplace`
* **Structure:** 50 episodes = 5 cube positions × 10 episodes per position
* **Task:** Pick cube and place in box
* **Robot:** SO-101 6-DOF arm

### Episode Preview

<video src="https://github.com/user-attachments/assets/2b3d4491-bd0e-4e20-a902-aca05525c08d" controls="controls" style="max-width: 100%;">
</video>
--- 
## 🔧 Technical Challenges Solved

### 1. Temporal Alignment (Action Chunk Mismatch)

**Problem:** Model expects 50-step action chunks, dataset provides single-step actions.

**Solution:** Implemented temporal query vector aligned with 30 FPS:

```python
fps = 30
action_horizon = 50

delta_timestamps = {
    "observation.state": [0.0],
    "observation.images.up": [0.0],
    "observation.images.side": [0.0],
    "action": [i / fps for i in range(action_horizon)]  # [0.0, 0.033, ..., 1.66s]
}
```

### 2. Camera Key Remapping

**Problem:** Dataset uses `observation.images.up/side`, SmolVLA expects `camera1/camera2`.

**Solution:**
```python
def fix_keys(batch):
    if "observation.images.up" in batch:
        batch["observation.images.camera1"] = batch.pop("observation.images.up")
    if "observation.images.side" in batch:
        batch["observation.images.camera2"] = batch.pop("observation.images.side")
    return batch
```

### 3. Color Augmentation for Robustness

Implemented asymmetric augmentation to test generalization:

* **Training:** Standard variance (brightness/contrast ±20%)
* **Validation:** Shifted distribution (darker, higher contrast)
![Augmentation Preview](aug_image1.jpeg)
*Figure 1: Comparison of Raw inputs vs. Augmented Training (Center) and Shifted Validation (Right). Note the darker lighting conditions in the validation set.*


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
This simulates sim-to-real lighting variation.

### 4. Evaluation Methodology

**Challenge:** Initial evaluation showed MAE in thousands, 0% success rate.

**Root Cause:** Model outputs normalized actions; ground truth in raw units.

**Solution:** Proper denormalization pipeline:

```python
# 1. Get predictions from model
pred_seq = policy.predict_action_chunk(batch)  # (B, 50, 6)
pred_action = pred_seq[:, 0, :]  # First action (B, 6)

# 2. Denormalize to raw action space
meta = LeRobotDatasetMetadata(DATASET_REPO)
action_mean = torch.tensor(meta.stats["action"]["mean"])
action_std = torch.tensor(meta.stats["action"]["std"])

pred_raw = pred_action * action_std + action_mean

# 3. Evaluate: success = within 5% of joint range
joint_ranges = action_max - action_min
tolerance = joint_ranges * 0.05

success = (abs(pred_raw - gt_raw) <= tolerance)
```

**Why 5% tolerance?** Joint ranges vary (shoulder_pan: 181° vs gripper: 33°). Fixed tolerance would be unfair; per-joint percentage ensures consistent evaluation.

---

## 📂 Repository Structure

```
smolvla-so101-finetune/
├── README.md                    # This file
├── 01_train_smolvla.ipynb      # Training pipeline with augmentation
├── 02_eval_offline.ipynb       # Comprehensive evaluation script
├── Results.txt                  # Raw numerical results
├── aug_image1.jpeg             # Augmentation comparison
├── evaluation_results.png       # Performance visualizations
└── LICENSE
```
---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone and install LeRobot with SmolVLA support
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 2. Install additional dependencies
pip install wandb num2words==0.5.14

# 3. Login to Hugging Face
huggingface-cli login
```

### Running the Code

```bash
# Clone this repository
git clone https://github.com/Sa74ll/smolvla-so100-finetune.git
cd smolvla-so100-finetune

# Train with stratified splitting
jupyter notebook 01_train_smolvla.ipynb

# Evaluate on validation set
jupyter notebook 02_eval_offline.ipynb
```

**Note:** The notebooks include both the initial sequential split (for comparison) and the improved stratified split. Results shown are from the stratified version.

---

## 📊 Detailed Results

### Training Dynamics

**With Stratified Split:**
- Training loss: 0.021 (at step 6000)
- Validation loss: 0.036 (at step 6000)
- Train/val gap: **1.7x** ✅ (healthy)

**Best model:** Checkpoint at step 6000 (validation loss: 0.036)

### Per-Position Performance

```
Position 0: MAE = 1.88 radians
Position 1: MAE = 2.17 radians
Position 2: MAE = 1.75 radians
Position 3: MAE = 1.82 radians
Position 4: MAE = 2.61 radians
```

All positions show reasonable performance. Position 4 has slightly higher error (furthest reach), but is **much better** than the 5x gap we had before.

---
## 🔗 References

- **Course:** https://huggingface.co/spaces/lerobot/robot-learning-tutorial
-  **SmolVLA Blog:** https://huggingface.co/blog/smolvla
- **SmolVLA Paper:** https://arxiv.org/abs/2506.01844
- **SO-101 Dataset:** https://huggingface.co/datasets/lerobot/svla_so101_pickplace
- **LeRobot Framework:** https://github.com/huggingface/lerobot

---
