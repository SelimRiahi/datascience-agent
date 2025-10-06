# Data Science Agent

A complete data science workflow tool with data cleaning, machine learning, and visualization. Built with Streamlit and powered by LLMs for intelligent recommendations.

## Features

- **Data Cleaning**: Upload messy datasets and get them cleaned automatically with AI recommendations
- **ML Training**: Train models with AutoML, compare algorithms, and track performance
- **Visualization**: Generate PowerBI-style dashboards with smart chart recommendations

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your API keys in `data_quality/secrets_config.py`:

```python
GEMINI_API_KEY = "your_gemini_key_here"
GROQ_API_KEY = "your_groq_key_here"
```

3. Run the app:

```bash
streamlit run app.py
```

## Usage

### Data Cleaning

Upload a CSV or Excel file, review the AI suggestions, and download the cleaned dataset.

### ML Training

Choose your prediction type (classification or regression), pick columns, and train models. Results are saved with history tracking.

### Visualization

Select your data columns and let the AI recommend the best visualizations. Generate full dashboards with statistics and PowerBI-style charts.

## Project Structure

```
├── app.py                          # Main Streamlit app
├── data_quality/                   # Data cleaning module
│   ├── data_cleaning_agent.py     # Cleaning logic
│   └── secrets_config.py          # API keys
├── ml_agent/                       # ML training module
│   ├── smart_ml_agent.py          # AutoML engine
│   └── ml_database.py             # Training history
├── visualization/                  # Visualization module
│   ├── smart_recommender.py       # Chart recommendations
│   ├── beautiful_dashboard.py     # Dashboard generator
│   └── learning_database.py       # Learning system
└── insights/                       # Analysis tools
    └── insights_agent.py          # Data insights
```

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly
- **AI**: Google Gemini, Groq LLaMA

## Notes

- Database files (`.db`) are created automatically
- Temp files are ignored by git
- Models and large datasets are not committed
