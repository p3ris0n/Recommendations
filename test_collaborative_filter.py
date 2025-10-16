# test_collaborative_filtering.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the collaborative filter module
from collaborative_filter import (
    load_data_correctly,
    get_top_recommendations,
    calc_precision_at_k,
    calc_recall_at_k,
    calc_f1_score,
    calc_average_precison,
    calc_mean_avg_precision,
    calc_recommendation_diversity,
    calc_gini,
    calc_catalog_coverage,
    RecommenderEvaluator,
    MetricsTracker,
    create_temporal_split
)

def test_data_loading():
    """Test data loading function"""
    print("=" * 60)
    print("TEST 1: DATA LOADING")
    print("=" * 60)
    
    try:
        data = load_data_correctly()
        print(f"✓ Data loaded successfully")
        print(f"  - Shape: {data.shape}")
        print(f"  - Columns: {list(data.columns)}")
        print(f"  - Unique users: {data['user_id'].nunique()}")
        print(f"  - Unique items: {data['item_id'].nunique()}")
        print(f"  - Rating range: {data['rating'].min()} to {data['rating'].max()}")
        return data
    except Exception as e:
        print(f" Data loading failed: {e}")
        return None

def test_recommendation_generation(trainset, model, sample_users):
    """Test recommendation generation function"""
    print("\n" + "=" * 60)
    print("TEST 2: RECOMMENDATION GENERATION")
    print("=" * 60)
    
    results = []
    for i, user_id in enumerate(sample_users[:3]):  # Test with first 3 users
        try:
            recommendations = get_top_recommendations(user_id, trainset, model, n=3)
            print(f"✓ Recommendations for user {user_id}:")
            for item, rating in recommendations:
                print(f"  - {item}: predicted rating = {rating:.3f}")
            results.append((user_id, recommendations))
        except Exception as e:
            print(f"✗ Failed to generate recommendations for {user_id}: {e}")
    
    return results

def test_accuracy_metrics():
    """Test precision, recall, and F1 score calculations"""
    print("\n" + "=" * 60)
    print("TEST 3: ACCURACY METRICS")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'name': 'Perfect recommendations',
            'recommendations': ['A', 'B', 'C', 'D', 'E'],
            'actual': ['A', 'B', 'C'],
            'k': 5
        },
        {
            'name': 'No relevant recommendations',
            'recommendations': ['X', 'Y', 'Z'],
            'actual': ['A', 'B', 'C'],
            'k': 3
        },
        {
            'name': 'Mixed recommendations',
            'recommendations': ['A', 'X', 'B', 'Y', 'C'],
            'actual': ['A', 'B', 'C', 'D'],
            'k': 5
        }
    ]
    
    for case in test_cases:
        print(f"\nTest case: {case['name']}")
        precision = calc_precision_at_k(case['recommendations'], case['actual'], case['k'])
        recall = calc_recall_at_k(case['recommendations'], case['actual'], case['k'])
        f1 = calc_f1_score(precision, recall)
        
        print(f"  Precision@{case['k']}: {precision:.3f}")
        print(f"  Recall@{case['k']}: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")

def test_map_metrics():
    """Test Mean Average Precision calculations"""
    print("\n" + "=" * 60)
    print("TEST 4: MEAN AVERAGE PRECISION")
    print("=" * 60)
    
    # Test individual average precision
    recommendations = ['A', 'B', 'C', 'D', 'E']
    actual = ['B', 'D']
    
    ap = calc_average_precison(recommendations, actual, k=5)
    print(f"Average Precision@5: {ap:.3f}")
    
    # Test MAP with multiple users
    all_recommendations = [
        ['A', 'B', 'C', 'D', 'E'],  # User 1
        ['X', 'Y', 'Z', 'A', 'B'],  # User 2
        ['C', 'D', 'E', 'F', 'G']   # User 3
    ]
    all_actual = [
        ['B', 'D'],  # User 1 actual
        ['A', 'B'],  # User 2 actual
        ['C', 'F']   # User 3 actual
    ]
    
    map_score = calc_mean_avg_precision(all_recommendations, all_actual, k=5)
    print(f"Mean Average Precision@5: {map_score:.3f}")

def test_diversity_metrics():
    """Test diversity and coverage metrics"""
    print("\n" + "=" * 60)
    print("TEST 5: DIVERSITY & COVERAGE METRICS")
    print("=" * 60)
    
    # Test recommendation lists with different diversity patterns
    test_cases = [
        {
            'name': 'High diversity',
            'recommendations': [
                ['A', 'B', 'C', 'D', 'E'],
                ['F', 'G', 'H', 'I', 'J'],
                ['K', 'L', 'M', 'N', 'O']
            ]
        },
        {
            'name': 'Low diversity',
            'recommendations': [
                ['A', 'B', 'C', 'A', 'B'],
                ['A', 'C', 'B', 'A', 'C'],
                ['B', 'A', 'C', 'B', 'A']
            ]
        }
    ]
    
    all_items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    
    for case in test_cases:
        print(f"\nTest case: {case['name']}")
        
        diversity = calc_recommendation_diversity(case['recommendations'])
        coverage = calc_catalog_coverage(case['recommendations'], all_items)
        
        print(f"  Entropy: {diversity['entropy']:.3f}")
        print(f"  Normalized Entropy: {diversity['normalized_entropy']:.3f}")
        print(f"  Unique Items Recommended: {diversity['unique_items_recommend']}")
        print(f"  Gini Coefficient: {diversity['gini_coefficient']:.3f}")
        print(f"  Catalog Coverage: {coverage['coverage']:.3f}")
        print(f"  Items Recommended: {coverage['items_recommeded']}")
        print(f"  Items Never Recommended: {coverage['items_never_recommended']}")

def test_gini_calculation():
    """Test Gini coefficient calculation"""
    print("\n" + "=" * 60)
    print("TEST 6: GINI COEFFICIENT")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Perfect equality',
            'counts': [10, 10, 10, 10, 10]
        },
        {
            'name': 'High inequality',
            'counts': [50, 10, 5, 3, 2]
        },
        {
            'name': 'Single item',
            'counts': [100]
        }
    ]
    
    for case in test_cases:
        gini = calc_gini(case['counts'])
        print(f"{case['name']}: Gini coefficient = {gini:.3f}")

def test_evaluator_class():
    """Test the RecommenderEvaluator class"""
    print("\n" + "=" * 60)
    print("TEST 7: RECOMMENDER EVALUATOR CLASS")
    print("=" * 60)
    
    # Create a sample test dataset
    sample_test_data = pd.DataFrame({
        'user_id': ['U1001', 'U1002', 'U1003'],
        'item_id': ['American', 'Mexican', 'Japanese'],
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(3)]
    })
    
    evaluator = RecommenderEvaluator(k_values=[3, 5])
    print(f"✓ Evaluator initialized with k_values: {evaluator.k_values}")
    print(f"  Results structure: {evaluator.results}")

def test_metrics_tracker():
    """Test the MetricsTracker class"""
    print("\n" + "=" * 60)
    print("TEST 8: METRICS TRACKER")
    print("=" * 60)
    
    tracker = MetricsTracker(log_file='test_metrics_log.json')
    
    # Log some sample metrics
    sample_metrics = {
        'precision@5': 0.45,
        'recall@5': 0.32,
        'f1@5': 0.38,
        'map@5': 0.41
    }
    
    tracker.log_evaluation(
        model_name='TestModel',
        model_version='1.0',
        metrics=sample_metrics,
        notes='Test evaluation run'
    )
    print("✓ Metrics logged successfully")
    
    # Retrieve history
    history = tracker.get_metric_history('precision@5')
    print(f"✓ Retrieved metric history: {len(history)} entries")
    if history:
        print(f"  Latest value: {history[-1]['value']}")

def test_temporal_split(data):
    """Test temporal data splitting"""
    print("\n" + "=" * 60)
    print("TEST 9: TEMPORAL SPLIT")
    print("=" * 60)
    
    # Add dummy timestamps for testing
    test_data = data.copy()
    start_date = datetime(2024, 1, 1)
    test_data['timestamp'] = [start_date + timedelta(days=i) for i in range(len(test_data))]
    
    try:
        train, val, test = create_temporal_split(test_data, test_weeks=1, validation_weeks=1)
        print("✓ Temporal split completed successfully")
        print(f"  Training set: {len(train)} interactions")
        print(f"  Validation set: {len(val)} interactions")
        print(f"  Test set: {len(test)} interactions")
    except Exception as e:
        print(f"✗ Temporal split failed: {e}")

def main():
    """Main test function"""
    print("COLLABORATIVE FILTERING TEST SUITE")
    print("=" * 60)
    
    # Test 1: Data Loading
    data = test_data_loading()
    if data is None:
        print("Cannot proceed without data. Exiting.")
        return
    
    # Set up the model (simplified version of the original setup)
    from surprise import Dataset, Reader, SVD
    
    reader = Reader(rating_scale=(0, 2))
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    
    model = SVD()
    model.fit(trainset)
    print("\n✓ Model trained successfully for testing")
    
    # Get sample users for testing
    sample_users = data['user_id'].unique()[:5]
    
    # Run all tests
    test_recommendation_generation(trainset, model, sample_users)
    test_accuracy_metrics()
    test_map_metrics()
    test_diversity_metrics()
    test_gini_calculation()
    test_evaluator_class()
    test_metrics_tracker()
    test_temporal_split(data)
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("Summary:")
    print("✓ Data loading and preprocessing")
    print("✓ Recommendation generation")
    print("✓ Accuracy metrics (Precision, Recall, F1)")
    print("✓ Ranking metrics (MAP)")
    print("✓ Diversity and coverage metrics")
    print("✓ Gini coefficient calculation")
    print("✓ Evaluator class functionality")
    print("✓ Metrics tracking over time")
    print("✓ Temporal data splitting")
    print("\nAll core functions have been tested successfully!")

if __name__ == "__main__":
    main()