# RAGAF-Diffusion: Dual-Stage Controllable Diffusion with Region-Adaptive Graph-Attention Fusion

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion?style=social)](https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion)

**Bridging Structure and Semantics: A Novel Diffusion Framework for Controllable Sketch-to-Image Generation**

</div>

---

## The Core Concept

**The Challenge:** How can we generate photorealistic images from sketches while preserving structural fidelity AND incorporating rich semantic details from text descriptions?

**Traditional Approaches Fall Short:**
- **Pure text-to-image**: Struggles with precise spatial control
- **Simple sketch conditioning**: Lacks semantic understanding of different regions
- **Uniform fusion**: Treats all image regions equally, missing context-specific requirements

**Our Innovation:** RAGAF-Diffusion treats sketch regions as a **semantic graph**, where each region can be intelligently fused with relevant text semantics through **graph attention**, enabling **region-aware**, **context-sensitive** generation.

---

## Conceptual Overview

### The Problem We Solve

Imagine you sketch a **house with a tree and a car**. You want:
- The **house** to be a "Victorian mansion"
- The **tree** to be a "cherry blossom in spring"
- The **car** to be a "vintage red sports car"

**Traditional methods** apply the entire text prompt uniformly across the image, leading to:
-  Semantic confusion (tree features bleeding into the house)
-  Structure-semantic mismatch (text details violating sketch structure)
-  Poor controllability (can't target specific regions)

**RAGAF-Diffusion** solves this by:
1. **Automatically detecting regions** in your sketch (house, tree, car)
2. **Building a spatial graph** of region relationships
3. **Using graph attention** to determine which text tokens are relevant to each region
4. **Adaptively fusing** sketch structure and text semantics based on the denoising timestep

### The RAGAF Philosophy

```
 Sketch Structure →  Region Graph →  Text Semantics →  Photorealistic Image
 (What + Where)      (Relationships)    (How + Details)      (Structure + Beauty)
```

**Three Core Principles:**

1. **Region-Awareness**: Different parts of a sketch have different semantic needs
2. **Adaptive Fusion**: Balance between structure and semantics evolves during generation
3. **Graph Reasoning**: Spatial relationships matter for coherent image synthesis

---

## Technical Innovation: The RAGAF Framework

### 1.  **Dual-Stage Architecture Design**

Our pipeline separates **structural generation** from **semantic refinement** for better controllability:

#### **Stage 1: Sketch-Guided Coarse Generation**
```
Purpose: Establish global structure and layout
Method:  ControlNet-style sketch conditioning
Output:  Structure-preserving coarse image
```

**Why separate stages?**
-  **Focus**: Each stage optimizes for one objective (structure vs. semantics)
-  **Flexibility**: Can use different guidance strengths for different generation goals
-  **Quality**: Prevents structure-semantic conflicts during generation

#### **Stage 2: RAGAF Semantic Refinement**
```
Purpose: Add semantic details while preserving structure
Method:  Region-adaptive graph attention fusion
Output:  Photorealistic image with rich details
```

**The key insight**: Structure and semantics have different importance at different denoising timesteps!

---

### 2. **Region-Adaptive Graph-Attention Fusion (RAGAF)**

This is the **core innovation** of our framework. Let's break it down:

#### **Step 1: Automatic Region Extraction** 

Instead of treating sketches as monolithic images, we decompose them into **meaningful regions**:

```python
# Conceptual process
sketch → edge_detection → connected_components → regions
```

**Example:**
```
Input Sketch: [House with tree and car]

Detected Regions:
├─ Region 1: House structure (center-top)
├─ Region 2: Tree foliage (left)  
├─ Region 3: Tree trunk (left-bottom)
├─ Region 4: Car body (right)
└─ Region 5: Background (scattered)
```

**Features per region** (6D vector):
-  Centroid location (x, y) - normalized
-  Area and perimeter
-  Bounding box dimensions
-  Shape compactness measure

**Why automatic?** No manual annotation required! Works with any sketch.

#### **Step 2: Graph Construction** 

Regions aren't isolated - they have **spatial relationships**:

```
Graph G = (V, E) where:
- V (nodes) = Sketch regions with spatial features
- E (edges) = Relationships between regions
```

**Edge Types:**
1. **Adjacency**: Regions that touch or overlap
2. **Proximity**: K-nearest neighbors by centroid distance
3. **Containment**: Nested regions (e.g., window inside house)

**Example Graph:**
```
     [House]
      /    \
[Window]  [Door]
     
   [Tree] ←→ [Ground]
     
   [Car]  ←→ [Road]
```

**Why graphs?** 
- Captures **spatial context** (tree is next to house)
- Enables **relational reasoning** (car should match road style)
- Models **part-whole relationships** (windows belong to house)

#### **Step 3: Graph Attention Mechanism** 

Not all region relationships are equally important! We use **multi-head graph attention**:

```python
# Simplified concept
for each region i:
    attention_weights = softmax(Q_i @ K_neighbors / √d)
    updated_features_i = ∑ attention_weights * V_neighbors
```

**What this does:**
-  Each region "attends" to relevant neighboring regions
-  Learns which relationships matter (e.g., roof relates to walls)
-  Propagates information across the graph

**Example:** When generating a "Victorian house":
- The **roof** region attends strongly to **wall** regions → Maintains architectural consistency
- The **window** region attends to **house** region → Ensures windows match house style
- The **tree** region has **weak attention** to house → Can have independent style

#### **Step 4: Region-Text Cross-Attention** 

This is where **semantic control** happens:

```python
# For each region, compute attention with text tokens
attention_map[region_i, token_j] = relevance(region_i, token_j)
```

**The Magic:** Different text tokens influence different regions!

**Example with prompt: "A Victorian house with a cherry blossom tree"**

```
Text Token Attention Map:

"Victorian" → High attention to [House, Roof, Window] regions
            → Low attention to [Tree, Ground] regions

"cherry"    → High attention to [Tree foliage] region
            → Zero attention to [House] regions

"blossom"   → High attention to [Tree foliage] region
            → Low attention to [Tree trunk] region
```

**Why powerful?**
-  **Targeted semantics**: "Victorian" only affects the house
-  **No bleeding**: Tree style doesn't leak into house
-  **Fine control**: Different parts get different semantic guidance

#### **Step 5: Adaptive Fusion Weights** 

The **final innovation**: Fusion weights adapt based on **diffusion timestep**:

```python
α_sketch(t) = high when t is large  (early steps, noisy)
β_text(t)   = high when t is small  (late steps, denoised)

fused_features = α(t) * sketch_features + β(t) * text_features
```

**Intuition:**
- **Early timesteps** (t=1000 → 700): Image is very noisy
  -  **Strong sketch guidance** (α=0.8): Establish correct structure
  -  Weak text guidance (β=0.2): Don't add details yet
  
- **Middle timesteps** (t=700 → 300): Structure forming
  -  **Balanced guidance** (α=0.5, β=0.5): Refine both structure and semantics
  
- **Late timesteps** (t=300 → 0): Near final image
  -  **Strong text guidance** (β=0.8): Add rich semantic details
  -  Weak sketch guidance (α=0.2): Allow flexibility for realism

**Why timestep-aware?**
-  **Structure first**: Get the layout right before adding details
-  **Details later**: Add texture, color, style when structure is stable
-  **Smooth transition**: Gradual shift from structure to semantics

---

### 3.  **Complete RAGAF Forward Pass**

Putting it all together:

```python
def RAGAF_forward(sketch, text_prompt, timestep_t):
    # 1. Extract regions from sketch
    regions = extract_regions(sketch)  # → List of Region objects
    
    # 2. Build spatial graph
    graph = build_graph(regions)  # → Graph G = (V, E)
    # V = node features (N, 6)
    # E = edge_index (2, num_edges)
    
    # 3. Graph attention over regions
    region_features = graph_attention(
        node_features=graph.V,
        edge_index=graph.E
    )  # → (N, hidden_dim)
    # Each region's features updated with spatial context
    
    # 4. Text encoding
    text_embeddings = clip_encoder(text_prompt)  # → (77, 768)
    
    # 5. Region-text cross-attention
    region_text_features = cross_attention(
        query=region_features,      # (N, hidden_dim)
        key_value=text_embeddings   # (77, 768)
    )  # → (N, hidden_dim)
    # Each region gets relevant text semantics
    
    # 6. Adaptive fusion
    α, β = compute_fusion_weights(timestep_t, region_features)
    fused_features = α * sketch_features + β * region_text_features
    
    # 7. Inject into UNet for denoising
    denoised_latent = unet(noisy_latent, fused_features, timestep_t)
    
    return denoised_latent
```

**Information Flow:**
```
Sketch → Regions → Graph → Spatial Context → Text Relevance → Adaptive Fusion → Refined Image
  |         |        |            |                 |                 |              |
  |         |        |            |                 |                 |              └─> Structure + Semantics
  |         |        |            |                 |                 └─> Timestep-aware balance
  |         |        |            |                 └─> Region-specific text influence
  |         |        |            └─> Neighborhood awareness
  |         |        └─> Spatial relationships
  |         └─> Meaningful components
  └─> User input
```

---

## � Advantages Over Existing Methods

### Comparison with State-of-the-Art

| Capability | Stable Diffusion | ControlNet | Sketch-guided GAN | **RAGAF-Diffusion** |
|------------|------------------|------------|-------------------|---------------------|
| **Structural Control** | ❌ Weak | ✅ Strong | ✅ Strong | ✅ **Strong** |
| **Semantic Control** | ✅ Strong | ✅ Strong | ❌ Limited | ✅ **Strong** |
| **Region Awareness** | ❌ None | ❌ None | ❌ None | ✅ **Full** |
| **Adaptive Fusion** | ❌ No | ❌ No | ❌ No | ✅ **Yes** |
| **Spatial Reasoning** | ❌ No | ❌ No | ❌ No | ✅ **Graph-based** |
| **Timestep Awareness** | ⚠️ Fixed | ⚠️ Fixed | N/A | ✅ **Adaptive** |
| **Multi-region Text** | ❌ No | ❌ No | ❌ No | ✅ **Yes** |

### **Key Differentiators**

#### 1. **Region-Level Semantic Control** 
- **Others**: Apply text prompt uniformly to entire image
- **RAGAF**: Each region receives targeted semantic guidance
- **Benefit**: Prevents semantic bleeding, enables complex multi-object scenes

#### 2. **Graph-based Spatial Reasoning** 
- **Others**: Treat pixels/patches independently
- **RAGAF**: Model explicit spatial relationships via graph attention
- **Benefit**: Coherent multi-object layouts, context-aware generation

#### 3. **Adaptive Structure-Semantic Balance** 
- **Others**: Fixed fusion weights throughout generation
- **RAGAF**: Dynamic fusion adapts to denoising progress
- **Benefit**: Better structure preservation with rich semantic details

#### 4. **Automatic Region Discovery** 
- **Others**: Require manual masks or segmentation models
- **RAGAF**: Automatic region extraction from sketch
- **Benefit**: Zero manual annotation, works with any sketch

---

##  Theoretical Foundation

### **Problem Formulation**

Given:
- **Sketch** $S \in \mathbb{R}^{H \times W}$ (edge map)
- **Text prompt** $T$ (natural language description)

Goal: Generate image $I \in \mathbb{R}^{3 \times H \times W}$ that:
1. Preserves spatial structure from $S$
2. Incorporates semantic details from $T$
3. Is photorealistic and coherent

### **Mathematical Framework**

#### **1. Region Extraction**
```
R = {r₁, r₂, ..., rₙ} = ConnectedComponents(S)
```
Where each region $r_i$ has features:
```
f_i = [x_i, y_i, area_i, perimeter_i, bbox_i, compactness_i] ∈ ℝ⁶
```

#### **2. Graph Construction**
```
G = (V, E) where:
- V = {f₁, f₂, ..., fₙ} (node features)
- E = {(i,j) | adjacency(rᵢ, rⱼ) ∨ proximity(rᵢ, rⱼ)}
```

#### **3. Graph Attention**
```
h'ᵢ = ∑ⱼ∈N(i) αᵢⱼ · Wᵥhⱼ

where: αᵢⱼ = softmax(eᵢⱼ)
       eᵢⱼ = LeakyReLU(aᵀ[Wₕhᵢ || Wₕhⱼ])
```

#### **4. Region-Text Cross-Attention**
```
Attention(Q, K, V) = softmax(QKᵀ/√d) · V

where: Q = region features (N × d)
       K, V = text embeddings (77 × 768)
```

#### **5. Adaptive Fusion**
```
α(t) = σ(w_α · φ(t) + b_α)  # Structure weight
β(t) = σ(w_β · φ(t) + b_β)  # Semantic weight

F_fused = α(t) ⊙ F_sketch + β(t) ⊙ F_text

where: φ(t) = timestep embedding
       σ = sigmoid activation
       ⊙ = element-wise product
```

#### **6. Denoising Objective**
```
L = 𝔼_t,x₀,ε [||ε - ε_θ(x_t, F_fused, t)||²]

where: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
       ε ~ N(0, I)
       ε_θ = UNet denoiser
```

### **Why This Works**

1. **Graph structure** captures spatial dependencies → Coherent layouts
2. **Cross-attention** aligns text with relevant regions → Targeted semantics
3. **Adaptive fusion** balances objectives over time → Structure + details
4. **Diffusion process** generates high-quality images → Photorealism

---

## 🏗️ Architecture Diagram

<div align="center">

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INPUT: Sketch + Text Prompt                       │
│              "A Victorian house with cherry blossom tree"            │
└─────────────┬────────────────────────────────────────────────────────┘
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│ SKETCH ANALYSIS │  │  TEXT ENCODING   │
├─────────────────┤  ├──────────────────┤
│ • Edge Detect   │  │ • CLIP Encoder   │
│ • Connected     │  │ • Token Embeddings│
│   Components    │  │   (77, 768)      │
│ • Extract N=15  │  │                  │
│   regions       │  │                  │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         ▼                    │
┌─────────────────────────────┼──────────────────────┐
│    REGION GRAPH CONSTRUCTION│                      │
├─────────────────────────────┘                      │
│  Nodes: [House(r₁), Roof(r₂), Window(r₃), ...,    │
│          Tree(r₇), Trunk(r₈), Car(r₁₂)]           │
│                                                     │
│  Edges: House↔Roof, House↔Window, Tree↔Trunk, ... │
│                                                     │
│  Features: (x, y, area, perimeter, bbox, compact.) │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│              STAGE 1: SKETCH-GUIDED DIFFUSION             │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Sketch Encoder (ControlNet-style)                   │  │
│  │  • Multi-scale feature extraction                   │  │
│  │  • Preserves edge information                       │  │
│  └─────────────────┬───────────────────────────────────┘  │
│                    │                                       │
│  ┌─────────────────▼───────────────────────────────────┐  │
│  │ UNet with Sketch Conditioning                       │  │
│  │  • Timestep t = 1000 → 0                           │  │
│  │  • Denoising with sketch guidance                   │  │
│  └─────────────────┬───────────────────────────────────┘  │
│                    │                                       │
│                    │ Coarse Image I₁                       │
└────────────────────┼───────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────┐
│         STAGE 2: RAGAF SEMANTIC REFINEMENT                │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │          RAGAF ATTENTION MODULE                      │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────┐    │ │
│  │  │ (A) GRAPH ATTENTION (Region Dependencies)   │    │ │
│  │  │                                              │    │ │
│  │  │  For each region rᵢ:                        │    │ │
│  │  │    Attend to neighbors N(i)                 │    │ │
│  │  │    Learn: Which regions influence which?    │    │ │
│  │  │                                              │    │ │
│  │  │  Example:                                    │    │ │
│  │  │    • Roof(r₂) → [House(r₁), Window(r₃)]    │    │ │
│  │  │    • Tree(r₇) → [Trunk(r₈), Ground(r₁₀)]   │    │ │
│  │  │                                              │    │ │
│  │  │  Output: Context-aware region features      │    │ │
│  │  │          h'ᵢ = Σⱼ αᵢⱼ · hⱼ                   │    │ │
│  │  └──────────────────┬───────────────────────────┘    │ │
│  │                     │                                │ │
│  │  ┌──────────────────▼───────────────────────────┐   │ │
│  │  │ (B) REGION-TEXT CROSS-ATTENTION              │   │ │
│  │  │                                               │   │ │
│  │  │  For each region rᵢ:                         │   │ │
│  │  │    Attend to text tokens T = [t₁, ..., t₇₇] │   │ │
│  │  │    Learn: Which words apply to this region?  │   │ │
│  │  │                                               │   │ │
│  │  │  Example with "Victorian house, cherry tree":│   │ │
│  │  │                                               │   │ │
│  │  │    House(r₁)  ← "Victorian" (high attn)     │   │ │
│  │  │               ← "house" (high attn)          │   │ │
│  │  │               ← "cherry" (zero attn)         │   │ │
│  │  │                                               │   │ │
│  │  │    Tree(r₇)   ← "Victorian" (zero attn)      │   │ │
│  │  │               ← "cherry" (high attn)         │   │ │
│  │  │               ← "tree" (high attn)           │   │ │
│  │  │                                               │   │ │
│  │  │  Output: Text-enriched region features       │   │ │
│  │  │          z'ᵢ = Attention(h'ᵢ, T, T)          │   │ │
│  │  └──────────────────┬────────────────────────────┘   │ │
│  │                     │                                │ │
│  └─────────────────────┼────────────────────────────────┘ │
│                        │                                  │
│  ┌─────────────────────▼────────────────────────────────┐ │
│  │ (C) ADAPTIVE FUSION (Timestep-Aware)                 │ │
│  │                                                       │ │
│  │  Compute fusion weights based on timestep t:         │ │
│  │                                                       │ │
│  │  t=1000 (early, noisy):   α=0.8  β=0.2              │ │
│  │    → Strong sketch, weak text                        │ │
│  │    → Focus on structure                              │ │
│  │                                                       │ │
│  │  t=500 (middle):          α=0.5  β=0.5               │ │
│  │    → Balanced                                        │ │
│  │    → Refine both                                     │ │
│  │                                                       │ │
│  │  t=100 (late, clean):     α=0.2  β=0.8              │ │
│  │    → Weak sketch, strong text                       │ │
│  │    → Focus on semantic details                       │ │
│  │                                                       │ │
│  │  Fused = α(t) · Sketch + β(t) · Text-Region        │ │
│  └──────────────────────┬────────────────────────────────┘ │
│                         │                                  │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │ Inject into UNet for Refinement                      │  │
│  │  • Continue denoising with fused features            │  │
│  │  • Structure preserved, details enhanced             │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                  │
└─────────────────────────┼──────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   FINAL OUTPUT IMAGE    │
              │                         │
              │  ✓ Structure preserved  │
              │  ✓ Semantics applied    │
              │  ✓ Photorealistic       │
              │  ✓ Region-coherent      │
              └─────────────────────────┘
```

</div>

### � Component Details

#### **Region Extraction Pipeline**
```python
Input: Sketch S (H×W grayscale)
│
├─> Edge Detection (if needed)
│   └─> Output: Binary edge map
│
├─> Connected Components Analysis
│   └─> Output: N labeled regions
│
├─> Feature Extraction per region:
│   ├─> Centroid (x̄, ȳ)
│   ├─> Area (pixel count)
│   ├─> Perimeter (boundary length)
│   ├─> Bounding box (x_min, y_min, width, height)
│   └─> Compactness (4π·area/perimeter²)
│
└─> Output: Region list R = [r₁, r₂, ..., rₙ]
```

#### **Graph Construction Methods**

1. **Hybrid Graph** (default - best performance):
   ```python
   # Combine adjacency + KNN
   edges = adjacency_edges(regions) ∪ knn_edges(regions, k=5)
   ```

2. **Pure Adjacency**:
   ```python
   # Only touching/overlapping regions
   edges = {(i,j) | IoU(rᵢ, rⱼ) > 0}
   ```

3. **KNN Graph**:
   ```python
   # K-nearest by centroid distance
   edges = {(i,j) | j ∈ KNN(centroid_i, k)}
   ```

#### **RAGAF Module Architecture**

```python
RAGAFAttentionModule(
    node_dim=6,              # Region features
    text_dim=768,            # CLIP embeddings
    hidden_dim=512,          # Internal representation
    num_graph_layers=2,      # Graph attention depth
    num_attention_heads=8,   # Multi-head attention
    dropout=0.1
)
# Total parameters: ~4.08M
```

---

##  Intuitive Examples: How RAGAF Works

### Example 1: "Victorian House with Cherry Blossom Tree"

```
INPUT SKETCH:
┌────────────────────────┐
│    /\                  │
│   /  \      ╭─╮        │  ← House with triangular roof
│  /____\     │ │        │  ← Tree with foliage
│  │  │ │     ╰─╯        │  ← Windows and trunk
│  │  │ │      │         │
└────────────────────────┘

REGION EXTRACTION (Automatic):
r₁: House body (center rectangle)
r₂: Roof (top triangle)
r₃: Left window
r₄: Right window
r₅: Door
r₆: Tree foliage (circle)
r₇: Tree trunk (vertical line)

GRAPH STRUCTURE:
    r₂(Roof)
       ↓
r₃──→ r₁(House) ←──r₄
   (Windows)
       ↓
    r₅(Door)
    
r₆(Foliage)
       ↓
    r₇(Trunk)

TEXT PROMPT: "A Victorian mansion with a cherry blossom tree in spring"

REGION-TEXT ATTENTION MAP:
┌──────────┬──────────┬─────────┬────────┬────────┬─────────┐
│ Region   │Victorian │mansion  │cherry  │blossom │spring   │
├──────────┼──────────┼─────────┼────────┼────────┼─────────┤
│House(r₁) │ ████████ │████████ │░░░░░░░░│░░░░░░░░│░░░░░░░ │
│Roof(r₂)  │ ████████ │████████ │░░░░░░░░│░░░░░░░░│░░░░░░░ │
│Window(r₃)│ ██████░░ │██████░░ │░░░░░░░░│░░░░░░░░│░░░░░░░ │
│Foliage(r₆)│░░░░░░░░│░░░░░░░░ │████████│████████│████████│
│Trunk(r₇) │░░░░░░░░ │░░░░░░░░ │████░░░░│░░░░░░░░│██░░░░░░│
└──────────┴──────────┴─────────┴────────┴────────┴─────────┘
█ = High attention  ░ = Low attention

RESULT:
 House receives "Victorian mansion" style → Ornate architecture
 Tree receives "cherry blossom spring" details → Pink flowers
 NO semantic bleeding → Tree stays floral, house stays architectural
```

### Example 2: Adaptive Fusion Over Time

```
TEXT: "A vintage red sports car"

TIMESTEP t=1000 (Early - Very Noisy):
α(sketch) = 0.85, β(text) = 0.15
┌─────────────────────────┐
│ ░░░▓▓░░░░░░░░░░░░       │  Focus: Get car SHAPE right
│ ░▓▓▓▓▓▓░░░░░░░░░░       │  → Sketch dominates
│ ▓▓▓▓▓▓▓▓░░░░░░░░        │  → Establish structure
│ ░▓▓▓▓▓░░░░░░░░░         │  Text influence: Minimal
└─────────────────────────┘

TIMESTEP t=500 (Middle - Partially Denoised):
α(sketch) = 0.5, β(text) = 0.5
┌─────────────────────────┐
│    ╔════╗               │  Focus: Refine BOTH
│ ╔══╬════╬══╗            │  → Balanced fusion
│ ║  ║○  ○║  ║            │  → Structure + details
│ ╚══════════╝            │  Car shape + vintage hints
└─────────────────────────┘

TIMESTEP t=100 (Late - Nearly Clean):
α(sketch) = 0.15, β(text) = 0.85
┌─────────────────────────┐
│    ╔════╗               │  Focus: Semantic DETAILS
│ ╔══╬════╬══╗            │  → Text dominates
│ ║🔴║ ◉◉ ║🔴║            │  → Add: Red color
│ ╚══════════╝            │  → Add: Vintage chrome
└─────────────────────────┘  → Add: Sports styling

FINAL OUTPUT:
 Structure preserved (car shape from sketch)
 Semantics applied (red, vintage, sports details)
 Photorealistic (smooth fusion)
```

### Example 3: Multi-Region Coherence

```
SKETCH: Living room scene

REGIONS:
r₁: Sofa (left)
r₂: Coffee table (center)
r₃: Lamp (right)
r₄: Window (background)
r₅: Floor

TEXT: "A modern minimalist living room with natural wood furniture"

GRAPH ATTENTION BENEFITS:

Without Graph Attention (Independent):
❌ Sofa: Modern style
❌ Table: Random wood type (oak)
❌ Lamp: Different metal (brass)
❌ Overall: Incoherent mixture

With Graph Attention (RAGAF):
✅ Sofa: Modern minimalist → Influences neighbors
✅ Table: Adopts same wood type (walnut) via attention to sofa
✅ Lamp: Coordinates metal finish via attention to table
✅ Floor: Harmonizes with furniture via global attention
✅ Overall: Coherent, unified aesthetic

GRAPH ATTENTION MECHANISM:
    Window(r₄)
       ↓
Lamp(r₃)  ⟷  Sofa(r₁)  ⟷  Table(r₂)
       ↘       ↓       ↙
          Floor(r₅)

Each region "sees" its neighbors and adjusts its features for coherence!
```

---


##  Implementation & Practical Usage

### Sketchy Dataset (Primary)  **VALIDATED & READY**

<div align="center">

| Metric | Value |
|--------|-------|
|  **Total Pairs** | **75,481** |
|  **Categories** | **125 objects** |
|  **Train** | 52,514 samples (70%) |
|  **Validation** | 11,532 samples (15%) |
|  **Test** | 11,435 samples (15%) |
|  **Size** | ~10 GB |

</div>

**Download:** [https://sketchy.eye.gatech.edu/](https://sketchy.eye.gatech.edu/)

**Categories Include:** airplane, apple, bear, bicycle, cat, dog, elephant, guitar, horse, house, motorcycle, penguin, piano, rabbit, shoe, tree, and 109 more!

### MS COCO (Secondary - Optional)

- **Purpose**: Multi-object complex scenes
- **Size**: ~25 GB (images + annotations)
- **Train**: 118,287 images
- **Val**: 5,000 images
- **Features**: 5 captions per image, auto-generated sketches

**Download:** [https://cocodataset.org/](https://cocodataset.org/)

>  **Note**: You can train on **Sketchy only**. COCO is optional for multi-object experiments.

---

##  Quick Start

### ⚡ Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python verify_dataset.py  # If you have datasets downloaded
```

###  Dataset Setup (Optional - for training)

**Option 1: Automatic Setup (Sketchy)**
```bash
# Download Sketchy dataset from https://sketchy.eye.gatech.edu/
# Extract to your preferred location

# Set environment variable
export SKETCHY_ROOT=/path/to/sketchy
echo 'export SKETCHY_ROOT=/path/to/sketchy' >> ~/.zshrc

# Verify dataset
python check_sketchy_format.py /path/to/sketchy
```

**Option 2: Detailed Guide**

See **[DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md)** for comprehensive instructions including:
- Step-by-step download instructions
- Directory structure requirements
- Validation scripts
- Troubleshooting tips

###  Verify Setup

```bash
# Run comprehensive validation
python verify_dataset.py

# Expected output:
# ✅ SKETCHY_ROOT: /path/to/sketchy
# ✅ Dataset loaded: 52,514 training samples
# ✅ ALL CHECKS PASSED - READY FOR TRAINING!
```

---

##  Training

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8+ | 12.1+ |
| GPU Memory | 16GB | 24GB+ |
| GPU | RTX 3090 | RTX 4090 / A100 |
| RAM | 32GB | 64GB+ |
| Storage | 50GB | 100GB+ |

>  **Mac Users**: Training on CPU is extremely slow. Use cloud GPU (RunPod, Lambda Labs, AWS).

###  Training Commands

**Quick Start (Development):**
```bash
# Train both stages on Sketchy dataset
python train.py --dataset sketchy

# Train with subset for quick testing
python train.py \
    --dataset sketchy \
    --categories airplane,apple,bear,cat,dog \
    --epochs 2
```

**Full Training:**
```bash
# Stage 1: Sketch-guided diffusion
python train.py \
    --stage stage1 \
    --dataset sketchy \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --checkpoint_dir ./checkpoints/stage1

# Stage 2: Semantic refinement
python train.py \
    --stage stage2 \
    --dataset sketchy \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --checkpoint_dir ./checkpoints/stage2

# Both stages (end-to-end)
python train.py \
    --stage both \
    --dataset sketchy \
    --batch_size 4 \
    --epochs 20
```

**Advanced Options:**
```bash
python train.py \
    --stage both \
    --dataset both \                    # Use both Sketchy and COCO
    --batch_size 8 \
    --gradient_accumulation_steps 2 \    # Effective batch size = 16
    --learning_rate 1e-4 \
    --mixed_precision fp16 \             # Memory efficient
    --use_lora \                         # LoRA fine-tuning
    --lora_rank 8 \
    --use_wandb \                        # Weights & Biases logging
    --wandb_project ragaf-diffusion \
    --seed 42
```

###  Cloud GPU Training (RunPod)

**Setup:**
```bash
# 1. Create RunPod account: https://runpod.io/
# 2. Select GPU: RTX 4090 or A100 recommended
# 3. SSH into instance

# 4. Clone and setup
git clone https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
pip install -r requirements.txt

# 5. Upload datasets to /workspace/datasets
# 6. Set environment variables
export SKETCHY_ROOT=/workspace/datasets/sketchy
export COCO_ROOT=/workspace/datasets/coco

# 7. Start training
python train.py \
    --stage both \
    --batch_size 8 \
    --mixed_precision fp16 \
    --checkpoint_dir /workspace/checkpoints \
    --use_wandb
```

**Expected Training Time:**

| Stage | Dataset | GPU | Epochs | Time |
|-------|---------|-----|--------|------|
| Stage 1 | Sketchy | RTX 4090 | 10 | ~6 hours |
| Stage 2 | Sketchy | RTX 4090 | 10 | ~8 hours |
| Both | Sketchy | RTX 4090 | 20 | ~14 hours |
| Both | Sketchy + COCO | A100 | 20 | ~24 hours |

---

##  Inference & Generation

### Basic Usage

```bash
python inference.py \
    --sketch examples/dog_sketch.png \
    --prompt "A photo of a golden retriever dog" \
    --stage1_checkpoint ./checkpoints/stage1/final.pt \
    --stage2_checkpoint ./checkpoints/stage2/final.pt \
    --output dog_output \
    --seed 42
```

### Advanced Options

```bash
python inference.py \
    --sketch my_sketch.png \
    --prompt "A beautiful sunset landscape with mountains" \
    --stage1_checkpoint checkpoints/stage1_best.pt \
    --stage2_checkpoint checkpoints/stage2_best.pt \
    --output landscape_output \
    --num_inference_steps 50 \          # More steps = higher quality
    --guidance_scale 7.5 \               # Classifier-free guidance
    --sketch_strength 0.8 \              # Sketch influence (0-1)
    --seed 42 \
    --save_intermediates                  # Save stage 1 output
```

### Output Structure

```
outputs/dog_output/
├── sketch.png              # Input sketch (normalized)
├── regions.png             # Extracted regions visualization
├── region_graph.png        # Graph structure visualization
├── stage1_output.png       # Stage 1 coarse output
├── stage2_output.png       # Stage 2 refined output (final)
├── comparison.png          # Side-by-side comparison
├── attention_maps.png      # Region-text attention visualization
└── metadata.json           # Generation parameters
```

### Batch Inference

```bash
# Generate from multiple sketches
python inference.py \
    --sketch_dir examples/sketches/ \
    --prompts_file examples/prompts.txt \
    --output_dir batch_outputs/ \
    --batch_size 4
```

---

##  Project Structure

```
Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/
│
├──  README.md                        # This file
├──  requirements.txt                 # Python dependencies
├──  DATASET_SETUP_GUIDE.md          # Detailed dataset instructions
├──  DEVELOPMENT.md                   # Developer documentation
├──  IMPLEMENTATION_SUMMARY.md        # Code organization details
│
├──  data/                            # Data processing
│   ├── sketch_extraction.py           # Edge detection (Canny, XDoG, HED)
│   ├── region_extraction.py           # Connected component analysis
│   ├── region_graph.py                # Spatial graph construction
│   └── __init__.py
│
├──  datasets/                        # Dataset loaders
│   ├── sketchy_dataset.py             # Sketchy dataset (75k pairs)
│   ├── coco_dataset.py                # MS COCO dataset
│   └── __init__.py
│
├──  models/                          # Core models
│   ├── ragaf_attention.py             # RAGAF module (4.08M params)
│   ├── adaptive_fusion.py             # Timestep-aware fusion
│   ├── stage1_diffusion.py            # Sketch-guided diffusion
│   ├── stage2_refinement.py           # Semantic refinement
│   └── __init__.py
│
├──  configs/                         # Configurations
│   ├── config.py                      # Training/inference configs
│   └── __init__.py
│
├──  utils/                           # Utilities
│   ├── common.py                      # Helper functions
│   └── __init__.py
│
├──  train.py                         # Main training script
├──  inference.py                     # Inference script
├──  verify_dataset.py                # Dataset validation
└──  check_sketchy_format.py          # Format checker
```



##  Examples

### Example 1: Simple Object
```bash
python inference.py \
    --sketch examples/apple_sketch.png \
    --prompt "A photo of a red apple on a wooden table" \
    --output apple_result
```

**Input Sketch** → **Stage 1 (Structure)** → **Stage 2 (Refined)**
```
   ┌─────┐          ┌─────┐              ┌─────┐
   │ ○○  │   ───►   │ Gray│      ───►    │Photo│
   │○  ○ │          │Apple│              │Apple│
   └─────┘          └─────┘              └─────┘
```

### Example 2: Animal with Details
```bash
python inference.py \
    --sketch examples/dog_sketch.png \
    --prompt "A golden retriever dog sitting on grass in a park" \
    --guidance_scale 8.0 \
    --sketch_strength 0.75
```

### Example 3: Complex Scene
```bash
python inference.py \
    --sketch examples/landscape_sketch.png \
    --prompt "A beautiful sunset over mountains with a lake in the foreground" \
    --num_inference_steps 100 \
    --sketch_strength 0.6
```

### Example 4: Multiple Variations
```bash
# Generate 5 variations from the same sketch
for seed in {1..5}; do
    python inference.py \
        --sketch examples/cat_sketch.png \
        --prompt "A fluffy white cat with blue eyes" \
        --output cat_var_$seed \
        --seed $seed
done
```

---

##  Configuration

### Default Configuration

Key hyperparameters in `configs/config.py`:

```python
# Model
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
hidden_dim = 512
num_graph_layers = 2
num_attention_heads = 8

# Training
learning_rate = 1e-4
batch_size = 4
stage1_epochs = 10
stage2_epochs = 10
mixed_precision = "fp16"

# Fusion
fusion_method = "learned"  # or "heuristic", "hybrid"
use_region_adaptive_fusion = True

# LoRA (efficient fine-tuning)
use_lora = True
lora_rank = 4
```

### Custom Configuration

**Method 1: Command Line**
```bash
python train.py \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --hidden_dim 768 \
    --num_graph_layers 3
```

**Method 2: YAML File**
```yaml
# config_custom.yaml
model:
  hidden_dim: 768
  num_graph_layers: 3
  fusion_method: "hybrid"

training:
  learning_rate: 5e-5
  batch_size: 8
  stage1_epochs: 15
  stage2_epochs: 15
```

```bash
python train.py --config config_custom.yaml
```

---

## Memory Optimization

For limited GPU memory:

```bash
# Use smaller batch size
--batch_size 2

# Use gradient accumulation
--gradient_accumulation_steps 4

# Use mixed precision
--mixed_precision fp16

# Freeze base UNet (train only RAGAF components)
--freeze_base_unet
```

## 📈 Monitoring & Logging

### Weights & Biases (Recommended)

```bash
# Enable W&B logging
python train.py --use_wandb --wandb_project ragaf-diffusion --wandb_run_name exp_001

# Tracked metrics:
# • Training/validation loss (stage 1 & 2)
# • Learning rate schedule
# • Fusion weights (sketch α vs text β)
# • Region-text attention maps
# • Generated sample images
# • GPU memory usage
# • Training speed (samples/sec)
```

**Dashboard Features:**
- 📊 Real-time loss curves
- 🖼️ Generated image samples every N epochs
- 🎯 Attention map visualizations
- ⚡ Training speed metrics
- 💾 Automatic model versioning

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir ./checkpoints --port 6006

# View at http://localhost:6006

# Logged data:
# • Scalars: Loss, LR, fusion weights
# • Images: Generated samples, attention maps
# • Histograms: Model gradients, activations
# • Graphs: Model architecture
```

### Console Logging

```bash
# Training progress display:
Epoch 1/10 [Stage 1] ━━━━━━━━━━━━━━━━━━━━ 100% 0:05:23
├── Train Loss: 0.1234
├── Val Loss: 0.1567
├── LR: 1.0e-04
├── Samples/sec: 12.5
└── ETA: 4h 32m

Epoch 1/10 [Stage 2] ━━━━━━━━━━━━━━━━━━━━ 100% 0:06:45
├── Train Loss: 0.0987
├── Val Loss: 0.1123
├── Sketch Weight (α): 0.35
├── Text Weight (β): 0.65
└── Attention Entropy: 2.45
```

---

##  Evaluation

### Metrics (Coming Soon)

We will provide evaluation scripts for:

**Image Quality:**
- **FID** (Fréchet Inception Distance) - Overall image quality
- **IS** (Inception Score) - Image diversity and quality
- **LPIPS** - Perceptual similarity

**Sketch Fidelity:**
- **Chamfer Distance** - Edge alignment with input sketch
- **IoU** - Region overlap with sketch regions
- **SSIM** - Structural similarity

**Text Alignment:**
- **CLIP Score** - Text-image semantic alignment
- **BERT Score** - Caption quality

**RAGAF-Specific:**
- **Attention Accuracy** - Region-text attention alignment
- **Fusion Balance** - Sketch vs text weight distribution
- **Graph Quality** - Region graph connectivity metrics

### Running Evaluation

```bash
# Coming soon
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --test_split test \
    --metrics fid,clip,chamfer \
    --output_dir evaluation_results/
```

---

### 💬 Getting Help

1. **Check Documentation:**
   - [DEVELOPMENT.md](DEVELOPMENT.md) - Architecture details
   - [DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md) - Dataset help

2. **Run Validation:**
   ```bash
   python verify_dataset.py
   ```

3. **GitHub Issues:**
   - Search existing issues
   - Create new issue with error logs

4. **Debug Mode:**
   ```bash
   python train.py --debug --verbose
   ```

---

### Related Work

**Diffusion Models:**
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., NeurIPS 2020)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., CVPR 2022)

**Controllable Generation:**
- [ControlNet](https://arxiv.org/abs/2302.05543) (Zhang et al., ICCV 2023)
- [T2I-Adapter](https://arxiv.org/abs/2302.08453) (Mou et al., 2023)

**Graph Attention:**
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., ICLR 2018)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **Stable Diffusion**: CreativeML Open RAIL-M License
- **HuggingFace Transformers**: Apache License 2.0
- **PyTorch**: BSD License
- **Sketchy Dataset**: Academic use only
- **MS COCO**: Creative Commons Attribution 4.0

---

##  Acknowledgments

This project builds upon excellent prior work:

- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)** by CompVis - Base diffusion model architecture
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** - Diffusion model framework and utilities
- **[ControlNet](https://github.com/lllyasviel/ControlNet)** by Lvmin Zhang - Inspiration for sketch conditioning
- **[Sketchy Dataset](https://sketchy.eye.gatech.edu/)** by Georgia Tech - Sketch-photo paired dataset
- **[MS COCO](https://cocodataset.org/)** - Image-caption dataset
- **PyTorch Team** - Deep learning framework

---


### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# 4. Run tests
pytest tests/

# 5. Format code
black .
flake8 .
```

---

<div align="center">


**Made with ❤️ by [Satyam Kumar](https://github.com/KumarSatyam24)**

</div>
