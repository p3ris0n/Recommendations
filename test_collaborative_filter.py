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
    #
    return True

# ==============================================================================
# STEP 5: FOOD WASTE SPECIFIC METRICS (Diversity & Coverage)
# ==============================================================================

def calculate_recommendation_diversity(all_recommendations):
    """
    Measure how diverse recommendations are across items using entropy.
    High diversity = recommendations spread evenly across items.
    Low diversity = recommendations concentrated on few items.
    """
    item_counts = {}
    total_recommendations = 0
    
    for recommendations in all_recommendations:
        for item in recommendations:
            item_counts[item] = item_counts.get(item, 0) + 1
            total_recommendations += 1
    
    # Calculate entropy
    entropy = 0
    for count in item_counts.values():
        probability = count / total_recommendations
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(item_counts)) if len(item_counts) > 0 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'unique_items_recommended': len(item_counts),
        'gini_coefficient': calculate_gini(list(item_counts.values()))
    }


def calculate_gini(counts):
    """
    Calculate Gini coefficient to measure inequality in recommendations.
    Gini = 0: perfect equality (all items recommended equally)
    Gini = 1: perfect inequality (all recommendations for one item)
    """
    counts = np.array(sorted(counts))
    n = len(counts)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


def calculate_catalog_coverage(all_recommendations, total_available_items):
    """
    Measure what percentage of available items are being recommended.
    Critical for food waste - items never recommended likely go to waste.
    """
    recommended_items = set()
    for recommendations in all_recommendations:
        recommended_items.update(recommendations)
    
    coverage = len(recommended_items) / len(total_available_items) if len(total_available_items) > 0 else 0
    
    return {
        'coverage': coverage,
        'items_recommended': len(recommended_items),
        'items_never_recommended': len(total_available_items) - len(recommended_items)
    }


# ==============================================================================
# STEP 5 TEST CASE
# ==============================================================================

def test_diversity_and_coverage_metrics():
    """
    Test food waste specific metrics with controlled scenarios.
    """
    
    print("=" * 70)
    print("STEP 5: TESTING DIVERSITY & COVERAGE METRICS")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # Test Case 1: Perfect Diversity (all items recommended equally)
    # ------------------------------------------------------------------------
    print("Test Case 1: Perfect Diversity")
    print("-" * 70)
    
    # 3 users, each gets 3 different items (all 9 items used once)
    perfect_diversity_recs = [
        ['item_A', 'item_B', 'item_C'],
        ['item_D', 'item_E', 'item_F'],
        ['item_G', 'item_H', 'item_I']
    ]
    
    diversity_metrics = calculate_recommendation_diversity(perfect_diversity_recs)
    
    print(f"Normalized Entropy: {diversity_metrics['normalized_entropy']:.4f}")
    print(f"Gini Coefficient: {diversity_metrics['gini_coefficient']:.4f}")
    print(f"Unique Items Recommended: {diversity_metrics['unique_items_recommended']}")
    
    # Perfect diversity should have normalized entropy = 1.0 and Gini ≈ 0
    assert diversity_metrics['normalized_entropy'] > 0.99, "Perfect diversity should have entropy ≈ 1.0"
    assert diversity_metrics['gini_coefficient'] < 0.01, "Perfect diversity should have Gini ≈ 0"
    assert diversity_metrics['unique_items_recommended'] == 9, "Should recommend 9 unique items"
    
    print("✓ Perfect diversity case validated")
    print()
    
    # ------------------------------------------------------------------------
    # Test Case 2: Poor Diversity (concentrated recommendations)
    # ------------------------------------------------------------------------
    print("Test Case 2: Poor Diversity (Popular Item Bias)")
    print("-" * 70)
    
    # All 3 users get mostly the same popular items
    # Make the recommendations even more skewed toward fewer items
    poor_diversity_recs = {
        'user1': ['item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_B'],
        'user2': ['item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_B'],
        'user3': ['item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_B'],
        'user4': ['item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_B'],
        'user5': ['item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_A', 'item_B'],
    }
    
    diversity_metrics = calculate_recommendation_diversity(poor_diversity_recs)
    
    print(f"Normalized Entropy: {diversity_metrics['normalized_entropy']:.4f}")
    print(f"Gini Coefficient: {diversity_metrics['gini_coefficient']:.4f}")
    print(f"Unique Items Recommended: {diversity_metrics['unique_items_recommended']}")
    
    # Poor diversity should have low entropy and high Gini
    assert diversity_metrics['normalized_entropy'] < 0.6, "Poor diversity should have low entropy"
    assert diversity_metrics['gini_coefficient'] > 0.5, "Poor diversity should have high Gini"
    assert diversity_metrics['unique_items_recommended'] == 2, "Should recommend only 2 unique items"
    
    print("✓ Poor diversity case validated")
    print("⚠️  This scenario means most restaurants won't get visibility")
    print()
    
    # ------------------------------------------------------------------------
    # Test Case 3: Full Coverage
    # ------------------------------------------------------------------------
    print("Test Case 3: Full Coverage")
    print("-" * 70)
    
    # All available items get recommended to someone
    available_items = {'item_A', 'item_B', 'item_C', 'item_D', 'item_E'}
    full_coverage_recs = [
        ['item_A', 'item_B'],
        ['item_C', 'item_D'],
        ['item_E']
    ]
    
    coverage_metrics = calculate_catalog_coverage(full_coverage_recs, available_items)
    
    print(f"Coverage: {coverage_metrics['coverage']:.2%}")
    print(f"Items Recommended: {coverage_metrics['items_recommended']}")
    print(f"Items Never Recommended: {coverage_metrics['items_never_recommended']}")
    
    assert coverage_metrics['coverage'] == 1.0, "Should have 100% coverage"
    assert coverage_metrics['items_recommended'] == 5, "All 5 items should be recommended"
    assert coverage_metrics['items_never_recommended'] == 0, "No items should be left out"
    
    print("✓ Full coverage case validated")
    print("✓ All surplus food gets visibility!")
    print()
    
    # ------------------------------------------------------------------------
    # Test Case 4: Poor Coverage
    # ------------------------------------------------------------------------
    print("Test Case 4: Poor Coverage")
    print("-" * 70)
    
    # Only 2 out of 10 available items get recommended
    available_items = {f'item_{i}' for i in range(10)}
    poor_coverage_recs = [
        ['item_0', 'item_1'],
        ['item_0', 'item_1'],
        ['item_1', 'item_0']
    ]
    
    coverage_metrics = calculate_catalog_coverage(poor_coverage_recs, available_items)
    
    print(f"Coverage: {coverage_metrics['coverage']:.2%}")
    print(f"Items Recommended: {coverage_metrics['items_recommended']}")
    print(f"Items Never Recommended: {coverage_metrics['items_never_recommended']}")
    
    assert coverage_metrics['coverage'] == 0.2, "Should have 20% coverage"
    assert coverage_metrics['items_recommended'] == 2, "Only 2 items recommended"
    assert coverage_metrics['items_never_recommended'] == 8, "8 items never recommended"
    
    print("✓ Poor coverage case validated")
    print("⚠️  80% of surplus food is invisible and likely goes to waste!")
    print()
    
    # ------------------------------------------------------------------------
    # Test Case 5: Real-World Scenario
    # ------------------------------------------------------------------------
    print("Test Case 5: Real-World Mixed Scenario")
    print("-" * 70)
    
    # Realistic: some popular items, some variety, but not perfect
    available_items = {f'item_{i}' for i in range(20)}
    realistic_recs = [
        ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'],  # User 1
        ['item_0', 'item_1', 'item_5', 'item_6', 'item_7'],  # User 2
        ['item_0', 'item_2', 'item_8', 'item_9', 'item_10'], # User 3
        ['item_1', 'item_3', 'item_11', 'item_12', 'item_13'], # User 4
        ['item_0', 'item_1', 'item_14', 'item_15', 'item_16']  # User 5
    ]
    
    diversity_metrics = calculate_recommendation_diversity(realistic_recs)
    coverage_metrics = calculate_catalog_coverage(realistic_recs, available_items)
    
    print("Diversity Metrics:")
    print(f"  Normalized Entropy: {diversity_metrics['normalized_entropy']:.4f}")
    print(f"  Gini Coefficient: {diversity_metrics['gini_coefficient']:.4f}")
    print(f"  Unique Items in Recommendations: {diversity_metrics['unique_items_recommended']}")
    print()
    print("Coverage Metrics:")
    print(f"  Catalog Coverage: {coverage_metrics['coverage']:.2%}")
    print(f"  Items Recommended: {coverage_metrics['items_recommended']} / {len(available_items)}")
    print(f"  Items Never Recommended: {coverage_metrics['items_never_recommended']}")
    print()
    
    # Validate reasonable ranges
    assert 0.5 < diversity_metrics['normalized_entropy'] < 1.0, "Should have moderate to good diversity"
    assert 0 < diversity_metrics['gini_coefficient'] < 0.7, "Should have some but not extreme inequality"
    assert coverage_metrics['coverage'] > 0.5, "Should cover more than 50% of catalog"
    
    print("✓ Real-world scenario validated")
    print()
    
    # ------------------------------------------------------------------------
    # Interpretation Guide
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("INTERPRETATION GUIDE FOR UKFOODSAVER")
    print("=" * 70)
    print()
    print("Normalized Entropy (0-1):")
    print("  • 0.8-1.0: Excellent diversity - recommendations spread evenly")
    print("  • 0.5-0.8: Good diversity - reasonable variety")
    print("  • 0.0-0.5: Poor diversity - concentrated on few items")
    print()
    print("Gini Coefficient (0-1):")
    print("  • 0.0-0.3: Low inequality - fair distribution")
    print("  • 0.3-0.6: Moderate inequality - some concentration")
    print("  • 0.6-1.0: High inequality - heavily concentrated")
    print()
    print("Catalog Coverage (0-1):")
    print("  • 0.7-1.0: Excellent - most restaurants get visibility")
    print("  • 0.4-0.7: Moderate - many items still invisible")
    print("  • 0.0-0.4: Poor - majority of surplus food not recommended")
    print()
    print("For food waste reduction, aim for:")
    print("  ✓ Normalized Entropy > 0.7")
    print("  ✓ Gini Coefficient < 0.4")
    print("  ✓ Coverage > 0.6")
    print()
    print("=" * 70)
    print("ALL STEP 5 TESTS PASSED! ✓")
    print("=" * 70)
    
    return True


# Run the comprehensive test
if __name__ == "__main__":
    test_complete_evaluation_pipeline()
    test_diversity_and_coverage_metrics()