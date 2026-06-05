# NutriScan

**AI-powered food intelligence and nutrition analysis platform.**
**Live Demo**: https://nutriscan-kuf6.onrender.com/

NutriScan combines computer vision, nutrition databases, health scoring, and ML-based recommendations in a modern Streamlit dashboard.

## Features

| Feature | Description |
|--------|-------------|
| Food Image Recognition | Upload a photo; AI predicts food category with confidence |
| Nutrition Intelligence | Calories, protein, carbs, fats, and natural-language summary |
| Health Score (0–100) | Weighted score from calories, protein ratio, fat, and balance |
| Goal-Based Suggestions | Weight loss, muscle gain, or healthy maintenance guidance |
| Similar Food Engine | scikit-learn similarity search for healthier alternatives |
| Streamlit Dashboard | SaaS-style UI with charts and actionable insights |

## Project structure

```
NutriScan/
├── app/
│   └── streamlit_app.py      # Main dashboard
├── src/
│   ├── preprocessing.py      # OpenCV image pipeline
│   ├── food_detector.py      # TensorFlow/Keras classifier
│   ├── nutrition_engine.py   # Nutrition DB lookup
│   ├── health_score.py       # 0–100 scoring engine
│   └── recommender.py        # Goals + similarity search
├── data/
│   └── nutrition_db.csv      # Per-food nutrition (extend here)
├── models/
│   ├── class_labels.json     # Class names for classifier
│   └── food_classifier.keras # (after training)
├── notebooks/
│   └── 01_explore_nutrition_data.ipynb
├── train_model.py            # Training pipeline
└── requirements.txt
```

## Quick start

### 1. Create environment

```bash
cd NutriScan
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

### 3. Demo vs production model

Without a trained model, **FoodDetector** runs in **demo mode** (deterministic pseudo-predictions from image stats). For production:

1. Organize images:

   ```
   data/train/<class_name>/*.jpg
   data/val/<class_name>/*.jpg
   ```

2. Train:

   ```bash
   python train_model.py --data-dir data --epochs 15
   ```

3. Restart Streamlit — it will load `models/food_classifier.keras` automatically.

## Extending the system

### Nutrition database

Add rows to `data/nutrition_db.csv` with columns:

`food_key`, `food_name`, `calories`, `protein_g`, `carbohydrates_g`, `fats_g`, `fiber_g`, `sugar_g`, `sodium_mg`, `serving_size_g`

Keep `food_key` aligned with classifier class folder names (e.g. `grilled_chicken`).

### Class labels

Update `models/class_labels.json` after training or when adding new food classes.

### Health score tuning

Adjust weights and thresholds in `src/health_score.py` (`HealthScoreEngine.WEIGHTS`).

## Tech stack

- Python 3.10+
- TensorFlow / Keras (MobileNetV2 transfer learning)
- OpenCV, NumPy, Pandas
- scikit-learn (NearestNeighbors similarity)
- Streamlit + Plotly

## License

MIT — use and modify for learning and production prototypes.
