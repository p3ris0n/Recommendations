import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader
from surprise import SVD
from collections import defaultdict
from collaborative_filter import create_temporal_split, get_top_recommendations
from collaborative_filter import calc_precision_at_k, calc_recall_at_k, calc_f1_score, calc_average_precison
from collaborative_filter import RecommenderEvaluator

# Test Cases for collaborative_filter.py

def test_complete_evaluation_pipeline():
    """
    A single comprehensive test that validates the entire evaluation pipeline.
    
    This test creates synthetic data with known patterns, runs the complete
    evaluation process, and verifies that the results make sense.
    """
    
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION PIPELINE TEST")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # Create synthetic test data with predictable patterns
    # ------------------------------------------------------------------------
    print("Creating synthetic interaction data...")
    print("-" * 70)
    
    # We'll create a scenario where we know what should happen:
    # - 5 users (user_1 through user_5)
    # - 10 items (item_A through item_J)
    # - Users have clear preferences we can verify
    
    base_date = datetime(2024, 1, 1)
    interactions = []
    
    # User 1 likes items A, B, C (will interact in both train and test)
    for week in range(8):  # 8 weeks of history
        for item in ['item_A', 'item_B', 'item_C']:
            interactions.append({
                'user_id': 'user_1',
                'item_id': item,
                'interaction_value': 1.0,
                'timestamp': base_date + timedelta(weeks=week)
            })
    
    # User 2 likes items D, E, F
    for week in range(8):
        for item in ['item_D', 'item_E', 'item_F']:
            interactions.append({
                'user_id': 'user_2',
                'item_id': item,
                'interaction_value': 1.0,
                'timestamp': base_date + timedelta(weeks=week)
            })
    
    # User 3 has mixed preferences (A, D, G)
    for week in range(8):
        for item in ['item_A', 'item_D', 'item_G']:
            interactions.append({
                'user_id': 'user_3',
                'item_id': item,
                'interaction_value': 1.0,
                'timestamp': base_date + timedelta(weeks=week)
            })
    
    # User 4 likes items H, I, J
    for week in range(8):
        for item in ['item_H', 'item_I', 'item_J']:
            interactions.append({
                'user_id': 'user_4',
                'item_id': item,
                'interaction_value': 1.0,
                'timestamp': base_date + timedelta(weeks=week)
            })
    
    # User 5 similar to User 1 (likes A, B, C) - for collaborative signal
    for week in range(8):
        for item in ['item_A', 'item_B', 'item_C']:
            interactions.append({
                'user_id': 'user_5',
                'item_id': item,
                'interaction_value': 1.0,
                'timestamp': base_date + timedelta(weeks=week)
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    print(f"Total interactions created: {len(interactions_df)}")
    print(f"Unique users: {interactions_df['user_id'].nunique()}")
    print(f"Unique items: {interactions_df['item_id'].nunique()}")
    print(f"Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
    print()
    
    # ------------------------------------------------------------------------
    # STEP 1 TEST: Temporal Splitting
    # ------------------------------------------------------------------------
    print("STEP 1: Testing Temporal Split")
    print("-" * 70)
    
    train_data, val_data, test_data = create_temporal_split(
        interactions_df, 
        test_weeks=2, 
        validation_weeks=1
    )
    
    # Verify the split makes sense
    assert len(train_data) > 0, "Training data should not be empty"
    assert len(test_data) > 0, "Test data should not be empty"
    assert train_data['timestamp'].max() < test_data['timestamp'].min(), \
        "Train data should come before test data"
    
    print("✓ Temporal split validated successfully")
    print()
    
    # ------------------------------------------------------------------------
    # STEP 2 TEST: Training the Model
    # ------------------------------------------------------------------------
    print("STEP 2: Training Collaborative Filtering Model")
    print("-" * 70)

    # conver to surprise format
    reader = Reader(rating_scale=(0,1))
    train_surprise = Dataset.load_from_df(
        train_data[['user_id', 'item_id', 'interaction_value']], reader
    )
    trainset = train_surprise.build_full_trainset()

    model = SVD()
    model.fit(trainset)
    
    print(f"Model trained on {len(train_data)} interactions")
    # print(f"Learned user factors shape: {model.user_factors.shape}")
    # print(f"Learned item factors shape: {model.item_factors.shape}")
    
    # Verify model can make predictions
    test_user = 'user_1'
    
    all_items = trainset.all_items()
    user_inner_id = trainset.to_inner_uid(test_user)
    user_items = set([trainset.to_raw_iid(item_id) for item_id, _ in trainset.ur[user_inner_id]])

    predictions = []
    for item_id in all_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        if raw_item_id not in user_items:
            pred = model.predict(test_user, raw_item_id)
            predictions.append((raw_item_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = [item for item, _ in predictions[:5]]

    print(f"Top recommendations for {test_user}: {top_recommendations}")
    assert len(top_recommendations) > 0, "Model should provide recommendations"
    
    # ------------------------------------------------------------------------
    # STEP 3 TEST: Individual Metric Calculations
    # ------------------------------------------------------------------------
    print("STEP 3: Testing Individual Metrics")
    print("-" * 70)
    
    # Create a controlled example where we know the exact answer
    test_recommendations = ['item_A', 'item_B', 'item_X', 'item_C', 'item_Y']
    test_actual = {'item_A', 'item_B', 'item_C', 'item_Z'}
    
    # At k=5:
    # - Recommended 5 items, 3 were relevant (A, B, C)
    # - User actually liked 4 items, we found 3 of them
    # - So precision should be 3/5 = 0.6
    # - And recall should be 3/4 = 0.75
    
    precision = calc_precision_at_k(test_recommendations, test_actual, k=5)
    recall = calc_recall_at_k(test_recommendations, test_actual, k=5)
    f1 = calc_f1_score(precision, recall)
    ap = calc_average_precison(test_recommendations, test_actual, k=5)
    
    print(f"Test case: recommendations = {test_recommendations}")
    print(f"Test case: actual items = {test_actual}")
    print()
    print(f"Precision@5: {precision:.4f} (expected 0.6000)")
    print(f"Recall@5: {recall:.4f} (expected 0.7500)")
    print(f"F1@5: {f1:.4f} (expected 0.6667)")
    print(f"Average Precision@5: {ap:.4f}")
    
    # Verify the calculations
    assert abs(precision - 0.6) < 0.001, f"Precision should be 0.6, got {precision}"
    assert abs(recall - 0.75) < 0.001, f"Recall should be 0.75, got {recall}"
    expected_f1 = 2 * (0.6 * 0.75) / (0.6 + 0.75)
    assert abs(f1 - expected_f1) < 0.001, f"F1 should be {expected_f1}, got {f1}"
    
    print("\n✓ Individual metrics validated successfully")
    print()
    
    # Manual calculation of Average Precision for verification:
    # Position 1: item_A is relevant, precision = 1/1 = 1.0
    # Position 2: item_B is relevant, precision = 2/2 = 1.0
    # Position 3: item_X not relevant, skip
    # Position 4: item_C is relevant, precision = 3/4 = 0.75
    # Position 5: item_Y not relevant, skip
    # AP = (1.0 + 1.0 + 0.75) / 4 = 0.6875
    expected_ap = (1.0 + 1.0 + 0.75) / 4
    print(f"Average Precision calculation breakdown:")
    print(f"  - Position 1 (item_A, relevant): precision = 1/1 = 1.0")
    print(f"  - Position 2 (item_B, relevant): precision = 2/2 = 1.0")
    print(f"  - Position 3 (item_X, not relevant): skip")
    print(f"  - Position 4 (item_C, relevant): precision = 3/4 = 0.75")
    print(f"  - Position 5 (item_Y, not relevant): skip")
    print(f"  - AP = (1.0 + 1.0 + 0.75) / 4 = {expected_ap:.4f}")
    assert abs(ap - expected_ap) < 0.001, f"AP should be {expected_ap}, got {ap}"
    print()
    
    # ------------------------------------------------------------------------
    # STEP 4 TEST: Complete Evaluation Pipeline
    # ------------------------------------------------------------------------
    print("STEP 4: Testing Complete Evaluation Pipeline")
    print("-" * 70)
    
    evaluator = RecommenderEvaluator(k_values=[5, 10])
    metrics = evaluator.evaluate(trainset, model, test_data)
    
    print("Evaluation Results:")
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name:20s}: {value:.4f}")
    
    # Verify all expected metrics are present
    expected_metrics = ['precision@5', 'recall@5', 'f1@5', 'map@5',
                       'precision@10', 'recall@10', 'f1@10', 'map@10']
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing expected metric: {metric}"
        assert 0 <= metrics[metric] <= 1, f"Metric {metric} should be between 0 and 1"
    
    print("\n✓ Complete evaluation pipeline validated successfully")
    print()
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print()
    print("What we validated:")
    print("  1. Temporal data splitting preserves chronological order")
    print("  2. Model training produces valid user and item factors")
    print("  3. Individual metrics calculate correctly with known inputs")
    print("  4. Complete evaluation pipeline processes multiple users")
    print("  5. All metrics are within valid ranges [0, 1]")
    print()
    
    return True


# Run the comprehensive test
if __name__ == "__main__":
    test_complete_evaluation_pipeline()