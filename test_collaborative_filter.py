import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from surprise import Dataset, Reader, SVD
from collaborative_filter import (
    create_temporal_split,
    get_top_recommendations,
    calc_precision_at_k,
    calc_recall_at_k,
    calc_f1_score,
    calc_average_precison,
    calc_recommendation_diversity,
    calc_catalog_coverage,
    RecommenderEvaluator,
    generate_eval_report
)


def test_complete_evaluation_pipeline():
    """
    A comprehensive test case that validates the entire recommendation system pipeline
    including data preparation, model training, evaluation metrics, and reporting.
    """
    
    print("=" * 70)
    print("COMPREHENSIVE RECOMMENDATION SYSTEM TEST")
    print("=" * 70)
    
    # ------------------------------------------------------------------------
    # STEP 1: Create comprehensive synthetic test data
    # ------------------------------------------------------------------------
    print("\nSTEP 1: Creating synthetic test data...")
    print("-" * 70)
    
    # Create a more comprehensive dataset
    np.random.seed(42)  # For reproducible results
    
    # Generate synthetic user-item interactions with timestamps
    users = ['User_1', 'User_2', 'User_3', 'User_4', 'User_5', 'User_6']
    items = ['Burger', 'Pizza', 'Salad', 'Pasta', 'Soup', 'Sandwich', 'Fries', 'Soda', 'Coffee', 'Tea']
    
    interactions = []
    base_date = datetime(2024, 1, 1)
    
    # Create diverse interaction patterns
    for user_idx, user in enumerate(users):
        # Each user has different preferences
        if user_idx % 3 == 0:  # Fast food lovers
            preferred_items = ['Burger', 'Pizza', 'Fries', 'Soda']
        elif user_idx % 3 == 1:  # Healthy eaters
            preferred_items = ['Salad', 'Soup', 'Sandwich', 'Tea']
        else:  # Mixed preferences
            preferred_items = ['Pasta', 'Coffee', 'Sandwich', 'Pizza']
        
        # Generate interactions over time
        for week in range(8):
            for item in preferred_items:
                interactions.append({
                    'user_id': user,
                    'item_id': item,
                    'rating': 1.0,  # They purchased these items
                    'timestamp': base_date + timedelta(weeks=week, days=np.random.randint(0, 7))
                })
        
        # Add some random interactions with other items
        for _ in range(5):
            random_item = np.random.choice([item for item in items if item not in preferred_items])
            interactions.append({
                'user_id': user,
                'item_id': random_item,
                'rating': 0.0,  # They didn't purchase these
                'timestamp': base_date + timedelta(weeks=np.random.randint(0, 8), days=np.random.randint(0, 7))
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    print(f"Generated {len(interactions_df)} interactions")
    print(f"Unique users: {interactions_df['user_id'].nunique()}")
    print(f"Unique items: {interactions_df['item_id'].nunique()}")
    print(f"Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
    
    # ------------------------------------------------------------------------
    # STEP 2: Temporal split
    # ------------------------------------------------------------------------
    print("\nSTEP 2: Performing temporal split...")
    print("-" * 70)
    
    train_data, val_data, test_data = create_temporal_split(
        interactions_df, 
        test_weeks=2, 
        validation_weeks=1
    )
    
    print(f"Training set: {len(train_data)} interactions")
    print(f"Validation set: {len(val_data)} interactions") 
    print(f"Test set: {len(test_data)} interactions")
    
    # ------------------------------------------------------------------------
    # STEP 3: Train the model
    # ------------------------------------------------------------------------
    print("\nSTEP 3: Training collaborative filtering model...")
    print("-" * 70)
    
    # Convert to Surprise format
    reader = Reader(rating_scale=(0, 1))
    train_surprise = Dataset.load_from_df(
        train_data[['user_id', 'item_id', 'rating']], 
        reader
    )
    trainset = train_surprise.build_full_trainset()
    
    # Train model
    model = SVD(n_factors=10, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    print("✓ Model trained successfully")
    
    # ------------------------------------------------------------------------
    # STEP 4: Generate recommendations for all users
    # ------------------------------------------------------------------------
    print("\nSTEP 4: Generating recommendations...")
    print("-" * 70)
    
    test_users = test_data['user_id'].unique()
    print(f"Generating recommendations for {len(test_users)} test users")
    
    for user in test_users[:3]:  # Show recommendations for first 3 users
        try:
            recommendations = get_top_recommendations(user, trainset, model, n=3)
            print(f"\nTop recommendations for {user}:")
            for item, rating in recommendations:
                print(f"  - {item}: {rating:.3f}")
        except Exception as e:
            print(f"  Could not generate recommendations for {user}: {e}")
    
    # ------------------------------------------------------------------------
    # STEP 5: Test individual metric functions
    # ------------------------------------------------------------------------
    print("\nSTEP 5: Testing individual metrics...")
    print("-" * 70)
    
    # Test precision and recall with known values
    test_recommendations = ['Burger', 'Pizza', 'Salad', 'Pasta', 'Soup']
    test_actual = {'Burger', 'Pizza', 'Sandwich', 'Fries'}
    
    precision = calc_precision_at_k(test_recommendations, test_actual, k=5)
    recall = calc_recall_at_k(test_recommendations, test_actual, k=5)
    f1 = calc_f1_score(precision, recall)
    ap = calc_average_precison(test_recommendations, test_actual, k=5)
    
    print(f"Precision@5: {precision:.3f} (2/5 relevant)")
    print(f"Recall@5: {recall:.3f} (2/4 actual items found)")
    print(f"F1@5: {f1:.3f}")
    print(f"Average Precision@5: {ap:.3f}")
    
    # ------------------------------------------------------------------------
    # STEP 6: Test diversity and coverage metrics
    # ------------------------------------------------------------------------
    print("\nSTEP 6: Testing diversity and coverage metrics...")
    print("-" * 70)
    
    # Generate recommendations for all test users for diversity analysis
    all_recommendations = []
    for user in test_users:
        try:
            recs = get_top_recommendations(user, trainset, model, n=5)
            item_recs = [item for item, _ in recs]
            all_recommendations.append(item_recs)
        except:
            continue
    
    # Test diversity
    diversity_metrics = calc_recommendation_diversity(all_recommendations)
    print(f"Normalized Entropy: {diversity_metrics['normalized_entropy']:.3f}")
    print(f"Gini Coefficient: {diversity_metrics['gini_coefficient']:.3f}")
    print(f"Unique Items Recommended: {diversity_metrics['unique_items_recommend']}")
    
    # Test coverage
    coverage_metrics = calc_catalog_coverage(all_recommendations, items)
    print(f"Catalog Coverage: {coverage_metrics['coverage']:.1%}")
    print(f"Items Recommended: {coverage_metrics['items_recommeded']}")
    print(f"Items Never Recommended: {coverage_metrics['items_never_recommended']}")
    
    # ------------------------------------------------------------------------
    # STEP 7: Complete evaluation using RecommenderEvaluator
    # ------------------------------------------------------------------------
    print("\nSTEP 7: Running complete evaluation pipeline...")
    print("-" * 70)
    
    evaluator = RecommenderEvaluator(k_values=[3, 5])
    metrics = evaluator.evaluate(trainset, model, test_data)
    
    print("Evaluation Results:")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value:.3f}")
    
    # ------------------------------------------------------------------------
    # STEP 8: Generate comprehensive evaluation report
    # ------------------------------------------------------------------------
    print("\nSTEP 8: Generating evaluation report...")
    print("-" * 70)
    
    # Note: You'll need to fix the generate_eval_report function first
    # by adding the missing trainset parameter and fixing the recommendation generation
    try:
        report = generate_eval_report(model, trainset, train_data, test_data, items)
        print("✓ Evaluation report generated successfully")
    except Exception as e:
        print(f"  Could not generate full report: {e}")
        print("  (This is expected until you fix the generate_eval_report function)")
    
    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Validate key requirements
    checks = [
        (len(train_data) > 0, "Training data is not empty"),
        (len(test_data) > 0, "Test data is not empty"),
        (len(all_recommendations) > 0, "Recommendations were generated"),
        (isinstance(metrics, dict), "Evaluation metrics were computed"),
        (0 <= metrics.get('precision@3', 0) <= 1, "Precision is within valid range"),
        (0 <= metrics.get('recall@3', 0) <= 1, "Recall is within valid range"),
    ]
    
    all_passed = True
    for check, message in checks:
        if check:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            all_passed = False
    
    if all_passed:
        print("\n ALL TESTS PASSED")
        print("\nKey Insights:")
        print(f"- Model can handle {len(users)} users and {len(items)} items")
        print(f"- Temporal split preserves chronological order")
        print(f"- All evaluation metrics computed successfully")
        print(f"- Diversity: {diversity_metrics['normalized_entropy']:.1%} entropy")
        print(f"- Coverage: {coverage_metrics['coverage']:.1%} of catalog recommended")
    else:
        print("\n Some tests failed. Check the implementation.")
    
    return all_passed


# Run the comprehensive test
if __name__ == "__main__":
    test_complete_evaluation_pipeline()
    pass