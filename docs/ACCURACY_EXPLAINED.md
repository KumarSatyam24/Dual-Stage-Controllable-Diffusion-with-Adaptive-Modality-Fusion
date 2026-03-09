# Stage 1 Accuracy Evaluation - Complete Guide

## 🎯 Quick Answer

**To find the accuracy of your Stage 1 model:**

```bash
python evaluate_stage1_accuracy.py --checkpoint /root/checkpoints/stage1/epoch_10.pth
```

This gives you a **0-100% accuracy score** plus detailed breakdown!

---

## 📊 What You Get

### Main Metrics Output:
```
🎯 OVERALL ACCURACY: 76.8%  ← Your main accuracy!

📐 Sketch Fidelity (Edge Consistency): 0.71 (71%)
   ↳ How well it follows the input sketch

✅ Category Accuracy: 82.0%  ← True Positive Rate
   ↳ % of images with correct category
   ↳ 82 correct, 18 wrong out of 100 samples
   
📈 Interpretation: ✅ GOOD - Model performs well!
```

---

## 🤔 True Positive & True Negative Explained

### For Generative Models:

#### **True Positive (TP)**
✅ **Model generates the CORRECT category**

Example:
- Input: Sketch of "cat" + prompt "a cat"  
- Generated: Image that looks like a cat
- CLIP classifies it as: "cat" ✅
- Result: **TRUE POSITIVE**

#### **False Positive (FP)**  
❌ **Model generates WRONG category or adds extra objects**

Example:
- Input: Sketch of "eyeglasses" + prompt "eyeglasses"
- Generated: Image of person wearing eyeglasses
- CLIP classifies it as: "person" ❌
- Result: **FALSE POSITIVE** (generated extra object)

#### **True Negative (TN)**
✅ **Model correctly AVOIDS unrelated categories**

Example:
- Input: Sketch of "cat"
- Generated: Image of cat
- CLIP does NOT classify it as: dog, car, airplane, etc. ✅
- Result: **TRUE NEGATIVE** for all non-cat categories

#### **False Negative (FN)**
❌ **Model FAILS to generate the target category**

Example:
- Input: Sketch of "cat" + prompt "a cat"
- Generated: Unclear/unrecognizable image
- CLIP classifies it as: "unknown" or wrong category ❌
- Result: **FALSE NEGATIVE**

---

## 📈 How Metrics Are Computed

### 1. Category Accuracy (True Positive Rate)
```python
Category Accuracy = (Correct predictions) / (Total samples)
                  = True Positives / (True Positives + False Positives)
```

**Example:**
- 100 test samples
- 82 classified correctly by CLIP → **82% accuracy**
- 18 classified incorrectly → 18% error

**This is your "True Positive Rate"!**

### 2. Edge Consistency (Sketch Fidelity)
```python
Edge Consistency = IoU(sketch_edges, generated_edges)
                 = Intersection / Union of edge pixels
```

**What it measures:**
- Extract edges from input sketch
- Extract edges from generated image  
- Compute overlap (Intersection over Union)
- Higher = better sketch following

### 3. Overall Accuracy
```python
Overall Accuracy = 0.5 × Edge Consistency + 0.5 × Category Accuracy
```

**Combines both:**
- 50% weight: Does it follow the sketch?
- 50% weight: Does it generate the right object?

---

## 🚀 Step-by-Step Usage

### Step 1: Install Dependencies
```bash
pip install git+https://github.com/openai/CLIP.git scikit-image opencv-python
```

### Step 2: Run Quick Evaluation (100 samples)
```bash
python evaluate_stage1_accuracy.py \
    --checkpoint /root/checkpoints/stage1/epoch_10.pth \
    --num_samples 100 \
    --guidance_scale 2.5
```

**Time:** ~5-10 minutes  
**Output:** Console results + JSON file

### Step 3: Interpret Results

**Example Output:**
```
🎯 OVERALL ACCURACY: 76.8%

📐 Edge Consistency: 0.71 (71%)
   Mean: 0.71, Std: 0.15
   Range: 0.34 - 0.92

✅ Category Accuracy:
   Top-1: 82.0%  ← 82 correct out of 100
   Top-5: 91.0%  ← Correct in top-5 for 91 samples

📈 Interpretation: ✅ GOOD
   - Follows sketches well (71%)
   - Generates correct objects 82% of the time
```

---

## 💡 What's a Good Score?

### Excellent (80-100%)
- ✅ Overall: >80%
- ✅ Edge Consistency: >0.75
- ✅ Category Accuracy: >85%

**Meaning:** Model works very well!

### Good (70-80%)
- ✅ Overall: 70-80%
- ✅ Edge Consistency: 0.65-0.75
- ✅ Category Accuracy: 75-85%

**Meaning:** Model performs well with minor issues

### Fair (60-70%)
- ⚠️ Overall: 60-70%
- ⚠️ Edge Consistency: 0.55-0.65
- ⚠️ Category Accuracy: 65-75%

**Meaning:** Model works but needs improvement

### Poor (<60%)
- ❌ Overall: <60%
- ❌ Edge Consistency: <0.55
- ❌ Category Accuracy: <65%

**Meaning:** Model needs significant improvement

---

## 🔬 Example Analysis

### Scenario 1: High Accuracy Model
```
Overall: 82%
Edge: 0.78
Category: 86%

Sample predictions:
✅ cat → cat (correct)
✅ airplane → airplane (correct)
✅ dog → dog (correct)
❌ eyeglasses → person (wrong - added person)
✅ chair → chair (correct)

Analysis:
- 86% correct category (86 TP, 14 FP)
- Strong sketch following
- Occasional extra objects
```

### Scenario 2: Needs Improvement
```
Overall: 64%
Edge: 0.58
Category: 70%

Sample predictions:
✅ cat → cat (correct)
❌ airplane → helicopter (wrong category)
❌ dog → wolf (close but wrong)
❌ eyeglasses → person (wrong - added person)
✅ chair → chair (correct)

Analysis:
- 70% correct category (70 TP, 30 FP)
- Mediocre sketch following
- Frequent category errors
```

---

## 🔄 Compare Different Settings

### Test Guidance Scale Impact:

```bash
# High guidance (text dominant)
python evaluate_stage1_accuracy.py \
    --checkpoint checkpoints/stage1/epoch_10.pth \
    --guidance_scale 7.5

# Optimal guidance (balanced)
python evaluate_stage1_accuracy.py \
    --checkpoint checkpoints/stage1/epoch_10.pth \
    --guidance_scale 2.5
```

**Expected:**
- guidance=7.5: Lower edge consistency, more extra objects
- guidance=2.5: Higher edge consistency, correct single objects

---

## 📁 Output Files

### Console Output
Pretty-printed results with interpretation

### JSON File: `stage1_evaluation_guidance{scale}.json`
```json
{
  "overall_accuracy": 76.8,
  "edge_consistency": {
    "mean": 0.71,
    "std": 0.15,
    "min": 0.34,
    "max": 0.92
  },
  "category_accuracy": {
    "top1": 0.82,
    "top5": 0.91
  },
  "num_samples": 100
}
```

---

## 🎓 Technical Details

### Why These Metrics?

**For generative models, we can't use traditional accuracy because:**
- No fixed "correct" output (many valid images for one sketch)
- Need to measure multiple aspects:
  1. **Fidelity**: Does it match the input sketch?
  2. **Semantic**: Does it generate the right object?
  3. **Quality**: Is the image good?

**Our approach:**
- Use CLIP (vision-language model) to classify generated images
- Compare classification with ground truth → True Positive Rate
- Measure edge overlap with sketch → Sketch Fidelity
- Combine into Overall Accuracy

### Why CLIP?

CLIP is trained on 400M image-text pairs and can:
- Classify images into categories (zero-shot)
- Measure text-image similarity
- Provide robust, human-like judgments

**It's like asking a human:** "What object is in this image?"

---

## ✅ Quick Checklist

**To evaluate your model:**

- [ ] Install CLIP: `pip install git+https://github.com/openai/CLIP.git`
- [ ] Install other deps: `pip install scikit-image opencv-python`
- [ ] Run evaluation: `python evaluate_stage1_accuracy.py --checkpoint <path>`
- [ ] Check Overall Accuracy (aim for >70%)
- [ ] Check Category Accuracy (your True Positive Rate)
- [ ] Check Edge Consistency (sketch fidelity)
- [ ] Save results JSON for records

---

## 🎯 Summary

**Main Question: How do I find accuracy?**
→ Run `evaluate_stage1_accuracy.py` - get 0-100% score

**Main Question: How do I find True Positives?**
→ Look at "Category Accuracy" (Top-1) - % of correct classifications

**Main Question: How do I find True Negatives?**
→ Implicitly measured (model avoids classifying as wrong categories)

**Simple interpretation:**
- **82% Category Accuracy** = 82% True Positive Rate
- Out of 100 samples: 82 correct ✅, 18 wrong ❌

**That's it! The script does everything automatically.** 🚀

---

## 📞 Need Help?

See these files:
- `ACCURACY_EVALUATION_GUIDE.md` - Detailed metric explanations
- `HOW_TO_MEASURE_ACCURACY.md` - Quick start guide
- `evaluate_stage1_accuracy.py` - Main script (well commented)

Or run:
```bash
python evaluate_stage1_accuracy.py --help
```
