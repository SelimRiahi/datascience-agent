# 🤖 Intelligent Data Science Agent

An AI-powered machine learning agent that learns from experience and provides intelligent model recommendations.

## 🎯 Features

- **🧠 Historical Intelligence**: Learns from past experiments to predict which models will work best
- **🔮 Performance Prediction**: Predicts model performance before training
- **⚡ Efficiency Optimization**: Can skip inferior models when confident
- **🎯 User Control**: You choose whether to trust the AI or test all models
- **📊 Interactive Interface**: Easy-to-use command line interface

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your data:**

   - Put your CSV files in the `data/` folder
   - The agent will let you choose which dataset to analyze

3. **Run the agent:**

   ```bash
   python main.py
   ```

4. **Follow the interactive prompts:**
   - Select your dataset
   - Choose target column (what to predict)
   - Choose features (what to use for prediction)
   - Decide whether to trust the AI's recommendations

## 💡 How It Works

### 🧠 Intelligence System

- **Database**: Stores results from every experiment in `ml_intelligence.db`
- **Pattern Recognition**: Identifies which models work best for similar datasets
- **Confidence Levels**: HIGH/MEDIUM/LOW confidence based on historical success

### 🎯 User Options

1. **Trust Agent** (⚡ 80% time savings): Train only the AI's recommended model
2. **Trust LLM**: Train the top 3 models suggested by the language model
3. **Full Analysis**: Test all available models for complete comparison

### 📈 Continuous Learning

- Every experiment is saved to build intelligence
- The more you use it, the smarter it becomes
- Accurately predicts performance before training

## 📊 Example Results

```
🧠 AI AGENT'S INTELLIGENCE:
   📚 Historical experience: 25 similar experiments
   🏆 Best performer: LinearRegression
   📈 Win rate: 36.0% (9/25 times)
   🔮 Predicted R²: 0.836 (±0.010)

🏆 FINAL RESULTS:
   🥇 Best model: LinearRegression
   📈 Score: 0.840
   🎯 AGENT ACCURACY: ✅ PERFECT!
   🔮 PERFORMANCE PREDICTION: 🎯 EXCELLENT
```

## 🔧 Project Structure

```
data_science_agent/
├── main.py                 # Main interface
├── data/                   # Put your CSV files here
├── ml_agent/              # Core AI agent code
│   ├── smart_ml_agent.py  # Main ML agent
│   └── ml_database.py     # Intelligence database
├── ml_intelligence.db     # AI's memory/learning database
└── requirements.txt       # Dependencies
```

## 🎉 Ready to Use!

This agent represents a complete evolution from basic ML pipeline to intelligent, autonomous data science agent that learns, predicts, and acts decisively based on accumulated intelligence.

**Start with `python main.py` and let the AI guide your machine learning experiments!** 🚀
