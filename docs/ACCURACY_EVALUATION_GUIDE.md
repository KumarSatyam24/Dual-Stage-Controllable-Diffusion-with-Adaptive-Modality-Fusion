# Stage 1 Model Accuracy Evaluation Guide

## Understanding Accuracy for Generative Models

Unlike classification models (which predict labels), **generative diffusion models create images**. So we need different metrics to measure "accuracy."

---

## 📊 Key Metrics Explained

### 1. **Overall Accuracy Score (0-100%)**
**What it measures:** Combined performance of the model  
**How it's computed:** 50% sketch fidelity + 50% category accuracy  
**Interpretation:**
- **80-100%**: Excellent - Model works very well
- **70-80%**: Good - Minor issues
- **60-70%**: Fair - Needs improvement  
- **<60%**: Poor - Significant problems

**This is your main "accuracy" metric!**

---

### 2. **Edge Consistency Score (0-1)**
**What it measures:** Does the generated image follow the input sketch?  
**How it's computed:** 
- Extract edges from generated image using Canny edge detection
- Compare with input sketch using IoU (Intersection over Union)
- IoU = (Matching edges) / (All edges)

**Example:**
```
Input sketch: Drawing of a cat
Generated image: Photo of a cat with similar outline
Edge Consistency: 0.75 (75% of edges match)
```

**Interpretation:**
- **>0.7**: Excellent sketch following
- **0.5-0.7**: Good sketch following
- **0.3-0.5**: Fair sketch following
- **<0.3**: Poor sketch following

---

### 3. **Top-1 Category Accuracy (0-100%)**
**This is like "True Positive Rate"!**

**What it measures:** Does the model generate the correct object category?  
**How it's computed:**
- Use CLIP (vision-language model) to classify generated image
- Check if CLIP's top prediction matches the true category
- Accuracy = (Correct predictions) / (Total samples)

**Example:**
```
Input: Sketch of "airplane" + prompt "an airplane"
Generated: Image of airplane
CLIP predicts: "airplane" ✅
True Positive!

Input: Sketch of "eyeglasses" + prompt "eyeglasses"  
Generated: Image with eyeglasses AND person
CLIP predicts: "person" ❌
False Positive (generated wrong category)
```

**Interpretation:**
- **>90%**: Excellent category accuracy
- **80-90%**: Good category accuracy
- **70-80%**: Fair category accuracy
- **<70%**: Poor category accuracy

---

### 4. **Top-5 Category Accuracy (0-100%)**
**What it measures:** Is the correct category in top 5 predictions?  
**More lenient metric** - useful to see if model is "close"

---

## 🤔 True Positive vs True Negative for Generative Models

Traditional classification has:
- **True Positive (TP)**: Correctly predicting positive class
- **True Negative (TN)**: Correctly predicting negative class
- **False Positive (FP)**: Wrongly predicting positive
- **False Negative (FN)**: Wrongly predicting negative

For generative models, we adapt this:

### **True Positive** in our context:
✅ **Model generates the correct category**
- Input: "cat" sketch + "a cat" prompt
- Output: Image that CLIP classifies as "cat"
- Result: **TRUE POSITIVE**

### **False Positive** in our context:
❌ **Model generates wrong category or adds extra objects**
- Input: "eyeglasses" sketch + "eyeglasses" prompt
- Output: Image that CLIP classifies as "person wearing eyeglasses"
- Result: **FALSE POSITIVE** (generated person instead of just eyeglasses)

### **True Negative** in our context:
✅ **Model correctly avoids generating unrelated categories**
- Input: "cat" sketch
- Output: Image is NOT classified as "dog", "car", "airplane", etc.
- Result: **TRUE NEGATIVE** for all non-cat categories

### **False Negative** in our context:
❌ **Model fails to generate the target category**
- Input: "cat" sketch + "a cat" prompt
- Output: Image that CLIP classifies as "tiger" or "unclear"
- Result: **FALSE NEGATIVE** (missed generating a cat)

---

## 📈 How to Interpret Results

### Example Result:
```
🎯 OVERALL ACCURACY: 75.50%

📐 Sketch Fidelity:
   Edge Consistency: 0.68 (68%)
   
✅ Category Accuracy:
   Top-1 Accuracy: 83.00%  ← This is your "True Positive Rate"
   Top-5 Accuracy: 92.00%
```

**What this means:**
- **75.5% Overall**: Good performance, but room for improvement
- **68% Edge Consistency**: Reasonably follows sketches, could be better
- **83% Top-1 Category Accuracy**: Out of 100 samples:
  - 83 samples: Generated correct category ✅ (TRUE POSITIVES)
  - 17 samples: Generated wrong category ❌ (FALSE POSITIVES/NEGATIVES)

---

## 🚀 Usage

### Basic Evaluation (100 samples):
```bash
python evaluate_stage1_accuracy.py --checkpoint checkpoints/stage1/epoch_10.pth
```

### More thorough evaluation (500 samples):
```bash
python evaluate_stage1_accuracy.py \
    --checkpoint checkpoints/stage1/epoch_10.pth \
    --num_samples 500
```

### Test different guidance scale:
```bash
python evaluate_stage1_accuracy.py \
    --checkpoint checkpoints/stage1/epoch_10.pth \
    --guidance_scale 3.0
```

---

## 📋 Output Files

The script generates:
- **Console output**: Pretty-printed results
- **JSON file**: `stage1_evaluation_guidance{scale}.json`
  - Contains all metrics
  - Sample predictions
  - Statistical details

---

## 💡 What Makes a Good Score?

### For Stage 1 (Sketch-Guided Generation):

**Excellent Performance:**
- Overall: >80%
- Edge Consistency: >0.7
- Category Accuracy: >85%

**Good Performance:**
- Overall: 70-80%
- Edge Consistency: 0.6-0.7
- Category Accuracy: 75-85%

**Needs Improvement:**
- Overall: <70%
- Edge Consistency: <0.6
- Category Accuracy: <75%

---

## 🔧 Requirements

Install additional dependencies:
```bash
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python scikit-image
```

---

## 📊 Comparison with Your Test Results

From your 125-category test with guidance_scale=7.5:
- ❌ Added extra objects (person with eyeglasses)
- ❌ Changed sketch structure
- ❌ Too detailed/realistic

**Expected with guidance_scale=2.5:**
- ✅ Better sketch fidelity (higher edge consistency)
- ✅ Correct single objects (higher category accuracy)
- ✅ Cleaner outputs

Run evaluation to quantify the improvement!

---

## 🎯 Next Steps

1. **Evaluate current model (guidance=7.5):**
   ```bash
   python evaluate_stage1_accuracy.py --guidance_scale 7.5
   ```

2. **Evaluate with optimal setting (guidance=2.5):**
   ```bash
   python evaluate_stage1_accuracy.py --guidance_scale 2.5
   ```

3. **Compare results:**
   - See which has higher overall accuracy
   - Check edge consistency improvement
   - Verify category accuracy increase

4. **Make decision:**
   - If guidance=2.5 is better → use it for all future inference
   - Document the optimal setting

---

## 📚 Additional Metrics (Advanced)

The comprehensive evaluation script (`evaluate_stage1_metrics.py`) includes:
- **FID (Fréchet Inception Distance)**: Distribution similarity
- **Inception Score**: Image quality and diversity
- **LPIPS**: Perceptual similarity
- **Structural Similarity (SSIM)**: Structure preservation

These require more computation but provide deeper insights.

---

## Summary

**To answer your question:**

1. **Model Accuracy** = Overall Accuracy Score (0-100%)
2. **True Positive** = Category Accuracy (correct category generated)
3. **True Negative** = Model avoids generating unrelated categories
4. **False Positive** = Wrong category or extra objects added
5. **False Negative** = Failed to generate target category

**Run the evaluation script to get quantitative accuracy metrics for your trained Stage 1 model!**
