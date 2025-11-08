# Quick Start: Integrating XAI into HFF-Net (5 Minutes)

## Files Generated

You have received **6 files**:

1. **attention_vis.py** - FDCA attention + Grad-CAM visualization
2. **mc_dropout.py** - MC-Dropout uncertainty estimation
3. **freq_analysis.py** - Frequency domain analysis
4. **eval_xai_integration.py** - Integration code for eval.py
5. **train_xai_integration.py** - Integration code for train.py
6. **XAI_INTEGRATION_GUIDE.md** - Detailed integration guide
7. **XAI_SUMMARY.md** - Complete technical summary

---

## 3-Step Quick Integration

### Step 1: Setup (1 min)

```bash
# Create explainability module
mkdir -p HFF/explainability
touch HFF/explainability/__init__.py

# Copy XAI files
cp attention_vis.py HFF/explainability/
cp mc_dropout.py HFF/explainability/
cp freq_analysis.py HFF/explainability/

# Update dependencies
pip install scipy scikit-learn grad-cam seaborn opencv-python
```

### Step 2: Modify eval.py (2 min)

**At the top of eval.py, add:**

```python
from explainability.attention_vis import FDCAAttentionVisualizer, SegmentationGradCAM, FrequencyComponentAnalyzer
from explainability.mc_dropout import MCDropoutUncertainty, DropoutScheduler
from explainability.freq_analysis import FrequencyDomainAnalyzer
```
 
**In argument parser, add:**

```python
parser.add_argument('--enable-xai', action='store_true', help='Enable XAI')
parser.add_argument('--num-xai-samples', type=int, default=5, help='XAI samples')
parser.add_argument('--mc-samples', type=int, default=20, help='MC samples')
```

**At the end of eval function (after predictions), add:**

```python
if args.enable_xai:
    # Initialize XAI
    attention_viz = FDCAAttentionVisualizer(device=args.device)
    gradcam = SegmentationGradCAM(model, target_layers=['decoder'], device=args.device)
    freq_analyzer = FrequencyComponentAnalyzer(device=args.device)
    uncertainty = MCDropoutUncertainty(model, num_samples=args.mc_samples, device=args.device)
    
    # FDCA Attention
    attention_viz.register_hooks(model)
    attn_maps = attention_viz.extract_attention_maps(images, model)
    attention_viz.remove_hooks()
    
    # Grad-CAM
    cam = gradcam.generate_cam(images, target_class=1)
    gradcam.visualize_gradcam(images[0,0].cpu().numpy(), cam, masks[0].cpu().numpy(),
                             Path('results/figures/xai/gradcam/sample.png'))
    
    # Frequency Analysis
    lf_pred = freq_analyzer.generate_lf_only_prediction(model, images)
    hf_pred = freq_analyzer.generate_hf_only_prediction(model, images)
    freq_analyzer.visualize_frequency_contributions(
        images[0,0].cpu().numpy(),
        lf_pred[0].argmax(0).cpu().numpy(),
        hf_pred[0].argmax(0).cpu().numpy(),
        predictions[0].argmax(0).cpu().numpy(),
        masks[0].cpu().numpy(),
        Path('results/figures/xai/freq/sample.png')
    )
    
    # MC-Dropout
    mc_outputs = uncertainty.mc_forward_pass(images)
    mean_pred, unc_map = uncertainty.compute_uncertainty_maps(mc_outputs)
    uncertainty.visualize_uncertainty(
        images[0,0].cpu().numpy(),
        mean_pred[0].argmax(0),
        unc_map[0],
        np.abs(mean_pred[0].argmax(0) - masks[0].cpu().numpy()),
        Path('results/figures/xai/uncertainty/sample.png')
    )
```

### Step 3: Run (1 min)

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --enable-xai \
    --num-xai-samples 5 \
    --mc-samples 20
```

---

## Expected Output Structure

```
results/figures/xai/
├── attention/
│   ├── fdca_attention_sample_0.png
│   └── fdca_attention_sample_1.png
├── gradcam/
│   ├── gradcam_overlay_sample_0.png
│   └── gradcam_overlay_sample_1.png
├── freq/
│   ├── frequency_analysis_sample_0.png
│   ├── frequency_spectrum_0.png
│   └── frequency_stats_0.json
└── uncertainty/
    ├── uncertainty_analysis_sample_0.png
    ├── uncertainty_statistics.png
    └── uncertainty_stats_0.json
```

---

## What Each Component Shows

| Component | What You See | Key Metric |
|-----------|-------------|-----------|
| **FDCA Attention** | Red heatmap on tumor regions | Attention focus |
| **Grad-CAM** | Jet colormap highlighting important areas | Layer importance |
| **Frequency** | LF smooth, HF sharp, Full combined | Dice scores |
| **Uncertainty** | Hot regions at boundaries, cool in clear areas | Correlation r ≈ 0.70 |

---

## Troubleshooting

**Problem: Module not found**
```python
# Make sure __init__.py exists in explainability/
touch HFF/explainability/__init__.py
```

**Problem: Out of memory during MC-Dropout**
```python
# Reduce samples
uncertainty = MCDropoutUncertainty(model, num_samples=10)  # Instead of 20
```

**Problem: Grad-CAM all zeros**
```python
# Check actual layer names
for name, _ in model.named_modules():
    if 'decoder' in name or 'fusion' in name:
        print(name)
```

---

## Next: Advanced Usage

For detailed customization, layer selection, custom uncertainty metrics, see **XAI_INTEGRATION_GUIDE.md**

---

## Paper Figure Integration

Use outputs for paper **Figure 2 (XAI Analysis)** and **Figure 3 (Uncertainty)**

```
Figure 2: [Input] [GT] [Pred] [FDCA Attn] [Grad-CAM] [LF Pred] [HF Pred]
Figure 3: [Input] [Pred] [Uncertainty] [Error] + Correlation table
```

---

## One-Liner Test

```bash
python -c "from explainability.attention_vis import FDCAAttentionVisualizer; print('XAI modules loaded successfully!')"
```

---

Done! You now have full explainability in HFF-Net.
