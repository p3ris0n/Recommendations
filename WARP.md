# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a collaborative filtering recommendation system built in Python, designed to provide personalized recommendations using SVD (Singular Value Decomposition). The system includes comprehensive evaluation metrics for measuring recommendation quality, diversity, and coverage.

## Core Architecture

### Main Components

**collaborative_filter.py** - The core module containing:
- `RecommenderEvaluator` class - Comprehensive evaluation framework for recommendation systems
- `MetricsTracker` class - Tracks evaluation metrics over time with JSON logging
- Core recommendation functions (`get_top_recommendations`, temporal data splitting)
- Evaluation metrics (precision@k, recall@k, F1, MAP, diversity, coverage)
- Specialized metrics for recommendation systems (Gini coefficient, catalog coverage, entropy)

**Data Structure Requirements:**
- Input data must have columns: `user_id`, `item_id`, `rating`
- Rating scale expected: 0-2 (configured in Reader)
- Timestamps required for temporal evaluation splits

### Key Classes and Functions

**Core Recommendation:**
- `get_top_recommendations(user_id, trainset, model, n=5)` - Generates top-N recommendations for a user
- `load_data_correctly()` - Loads and cleans CSV data with proper column mapping

**Evaluation Framework:**
- `RecommenderEvaluator` - Main evaluation class supporting multiple k-values (default: [5, 10, 20])
- `MetricsTracker` - Persistent metrics logging with timestamp tracking

**Temporal Evaluation:**
- `create_temporal_split(interactions_df, test_weeks=2, validation_weeks=1)` - Splits data by time for realistic evaluation

**Metrics:**
- Accuracy: `calc_precision_at_k`, `calc_recall_at_k`, `calc_f1_score`
- Ranking: `calc_average_precision`, `calc_mean_avg_precision`
- Diversity: `calc_recommendation_diversity`, `calc_gini`
- Coverage: `calc_catalog_coverage`

## Development Commands

### Running the System
```powershell
# Run main collaborative filter with sample data
python collaborative_filter.py

# Run comprehensive test suite
python test2.py

# Run unit tests
python test_collaborative_filter.py
```

### Data Requirements
- Place CSV data files in `data/` directory
- Expected files: `UKFS_testdata.csv` or `UKFSdatadata.csv`
- Excel support available via `openpyxl` (see test2.py)

### Testing
```powershell
# Install required dependencies first
pip install pandas numpy scipy scikit-learn scikit-surprise openpyxl

# Run the comprehensive test that covers all functions
python test2.py

# Run the unit test suite
python test_collaborative_filter.py
```

## Key Dependencies

- **pandas, numpy** - Data manipulation and numerical operations
- **scikit-surprise** - Collaborative filtering algorithms (SVD)
- **scipy** - Scientific computing (sparse matrices)
- **scikit-learn** - Machine learning utilities (TruncatedSVD)
- **openpyxl** - Excel file support (optional)

## Important Implementation Details

### Rating Scale Configuration
The system uses a 0-2 rating scale configured in the Surprise Reader. This is hardcoded and may need adjustment for different datasets.

### Evaluation Methodology
- **Temporal splits** are used instead of random splits for more realistic evaluation
- **Cold start users** (not in training data) are gracefully handled in evaluation
- **Multiple k-values** are evaluated simultaneously (typically 5, 10, 20)

### Metrics Interpretation
- **Precision@k**: Fraction of recommended items that are relevant
- **Recall@k**: Fraction of relevant items that are recommended  
- **MAP@k**: Mean Average Precision considering ranking quality
- **Diversity**: Measured via entropy and Gini coefficient
- **Coverage**: Percentage of catalog items that get recommended

### Data Handling
- Automatic column mapping from common formats (userID→user_id, food→item_id)
- Missing value handling and data type conversion
- Temporal timestamp generation for evaluation splits

## File Structure
```
/
├── collaborative_filter.py    # Main recommendation system
├── test2.py                  # Comprehensive test suite
├── test_collaborative_filter.py  # Unit tests
├── data/
│   ├── UKFS_testdata.csv     # Main dataset
│   └── UKFSdatadata.csv     # Alternative dataset
└── *.json                    # Metrics logs (generated)
```

## Working with the Evaluation Framework

### Running Full Evaluation Pipeline
The system includes `run_compile_eval_pipeline()` for end-to-end evaluation:
1. Temporal data splitting
2. Model training
3. Comprehensive evaluation
4. Report generation
5. Metrics logging

### Metrics Logging
- All evaluations are automatically logged to JSON files
- Historical tracking available via `MetricsTracker.get_metric_history()`
- Timestamps and model versions tracked

### Recommendation Quality Assessment
The `generate_eval_report()` function provides actionable insights:
- Flags low precision/recall with specific thresholds
- Identifies diversity issues and suggests solutions
- Highlights coverage problems and remediation strategies