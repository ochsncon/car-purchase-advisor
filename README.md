# AI Car Purchase Advisor

**Live Demo:** [huggingface.co/spaces/ochsncon/CarPurchaseAdvisor](https://huggingface.co/spaces/ochsncon/CarPurchaseAdvisor)

---

## What It Does

A single-page Gradio app that guides a user from a car photo to a personalised purchase assessment, combining three AI blocks:

1. **Computer Vision** — ResNet-18 (fine-tuned, HF Transformers) classifies the uploaded vehicle image into one of 19 car brands.
2. **ML Numeric Data** — RandomForestRegressor (scikit-learn) estimates the market price in CHF based on brand, engine power and fuel type.
3. **NLP** — GPT-4o-mini (OpenAI) generates a short, safety-constrained purchase assessment including budget check and financing orientation. Falls back to a deterministic rule-based explanation when no API key is set.

## AI Blocks & Data

| Block | Model | Training Data |
|---|---|---|
| Computer Vision | `microsoft/resnet-18`, 3 epochs, 19 classes | 2,519 train / 796 test images (`data/raw/Cars Dataset/`) |
| ML Numeric Data | `RandomForestRegressor`, n=300 | 1,175 rows from `data/raw/CarsDatasets2025.csv` |
| NLP | `gpt-4o-mini`, temperature=0.3 | No training, prompt engineering with runtime context |

## Project Structure

```
app.py                        # Gradio UI and pipeline orchestration
documentation.md              # Full project documentation
requirements.txt
src/
  data_preprocessing.py       # CSV loading, cleaning, feature engineering
  train_price_model.py        # ML training script
  train_vision_model.py       # CV fine-tuning script
  price_predictor.py          # ML inference
  vision_analyzer.py          # CV inference
  recommendation_engine.py    # NLP prompt logic and fallback
  config.py                   # Environment variable config
data/raw/
  CarsDatasets2025.csv        # Structured car listings
  Cars Dataset/               # Labeled car images (train/test)
models/
  price_model.pkl             # Trained RandomForest
  car-image-classifier/       # Fine-tuned ResNet-18 (HF format)
notebooks/
  ML_Numeric_Data_Block.ipynb # EDA and training walkthrough
```
