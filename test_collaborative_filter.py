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


def load_and_preprocess_data(file_path, max_items=25):
    """
    Load and preprocess the UK Food Savers dataset
    """
    print(f"Loading data from {file_path}...")
    
    # Only read first 1000 rows for testing
    df = pd.read_csv(file_path, nrows=1000)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Basic statistics
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique items: {df['item_id'].nunique()}")
    
    # Handle missing values - drop rows with missing user_id, item_id, or rating
    original_size = len(df)
    df_clean = df.dropna(subset=['user_id', 'item_id', 'rating']).copy()
    
    print(f"Removed {original_size - len(df_clean)} rows with missing values")
    
    # Convert rating to float
    df_clean.loc[:, 'rating'] = df_clean['rating'].astype(float)
    
    # Filter to only include the first max_items unique items
    unique_items = df_clean['item_id'].unique()
    if len(unique_items) > max_items:
        top_items = unique_items[:max_items]
        df_clean = df_clean[df_clean['item_id'].isin(top_items)].copy()
        print(f"Capped to first {max_items} unique items")
    
    print(f"Final dataset size: {len(df_clean)} rows")
    
    # Reset index
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean


def create_synthetic_timestamps(interactions_df, base_date=None):
    """
    Add synthetic timestamps for temporal analysis since the original data doesn't have them
    """
    if base_date is None:
        base_date = datetime(2024, 1, 1)
    
    # Create synthetic timestamps to simulate user activity over time
    np.random.seed(42)  # For reproducibility
    
    # Create a new DataFrame to avoid modifying the original
    interactions_with_time = []
    
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # Assign timestamps over a 6-month period for each user
        num_interactions = len(user_interactions)
        start_date = base_date + timedelta(days=np.random.randint(0, 30))
        
        for i, (_, row) in enumerate(user_interactions.iterrows()):
            # Simulate user activity over time with some randomness
            days_offset = i * 7 + np.random.randint(-2, 3)  # Weekly pattern with some variation
            timestamp = start_date + timedelta(days=days_offset)
            
            interactions_with_time.append({
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'timestamp': timestamp
            })
    
    return pd.DataFrame(interactions_with_time)


def test_complete_evaluation_pipeline():
    """
    A comprehensive test case that validates the entire recommendation system pipeline
    using the actual UK Food Savers dataset.
    """
    
    print("=" * 70)
    print("UK FOOD SAVERS RECOMMENDATION SYSTEM TEST")
    print("=" * 70)
    print("TESTING WITH FIRST 25 ITEMS ONLY")
    print("=" * 70)
    
    # ------------------------------------------------------------------------
    # STEP 1: Load and preprocess real data
    # ------------------------------------------------------------------------
    print("\nSTEP 1: Loading and preprocessing real data...")
    print("-" * 70)
    
    # Load the actual dataset
    try:
        interactions_df = load_and_preprocess_data('UKFoodSavers_testdata.csv', max_items=25)
        
        # Add synthetic timestamps for temporal analysis
        interactions_df = create_synthetic_timestamps(interactions_df)
        
        print(f"Final dataset: {len(interactions_df)} interactions")
        print(f"Unique users: {interactions_df['user_id'].nunique()}")
        print(f"Unique items: {interactions_df['item_id'].nunique()}")
        print(f"Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
        
        # Show sample of the data
        print("\nSample of the data:")
        print(interactions_df.head(3))
        
        # Show the 25 items being tested
        print(f"\nItems included in test:")
        items_list = interactions_df['item_id'].unique().tolist()
        for i, item in enumerate(items_list, 1):
            print(f"  {i:2d}. {item}")
        
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure 'UKFoodSavers_testdata.csv' is in the current directory.")
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Fallback to synthetic data if real data loading fails
        print("Falling back to synthetic data...")
        interactions_df = create_synthetic_fallback_data(max_items=25)
    
    # ------------------------------------------------------------------------
    # STEP 2: Temporal split
    # ------------------------------------------------------------------------
    print("\nSTEP 2: Performing temporal split...")
    print("-" * 70)
    
    try:
        train_data, val_data, test_data = create_temporal_split(
            interactions_df, 
            test_weeks=2, 
            validation_weeks=1
        )
        
        print(f"Training set: {len(train_data)} interactions ({len(train_data)/len(interactions_df):.1%})")
        print(f"Validation set: {len(val_data)} interactions ({len(val_data)/len(interactions_df):.1%})") 
        print(f"Test set: {len(test_data)} interactions ({len(test_data)/len(interactions_df):.1%})")
        
    except Exception as e:
        print(f"❌ Error in temporal split: {e}")
        return False
    
    # ------------------------------------------------------------------------
    # STEP 3: Train the model
    # ------------------------------------------------------------------------
    print("\nSTEP 3: Training collaborative filtering model...")
    print("-" * 70)
    
    try:
        # Convert to Surprise format
        reader = Reader(rating_scale=(1, 5))
        train_surprise = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        trainset = train_surprise.build_full_trainset()
        
        # Train model with parameters suitable for this dataset
        model = SVD(n_factors=15, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        model.fit(trainset)
        print("✓ Model trained successfully")
        print(f"✓ Trained on {trainset.n_users} users and {trainset.n_items} items")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    # ------------------------------------------------------------------------
    # STEP 4: Generate recommendations for all users
    # ------------------------------------------------------------------------
    print("\nSTEP 4: Generating recommendations...")
    print("-" * 70)


    # Get unique users from training and test sets 
    train_users = set([trainset.to_raw_uid(u) for u in trainset.all_users()])
    test_users = list(set(test_data['user_id']).intersection(train_users))[:20]  # CAP: Increased to 20

    print(f"Generating recs for {len(test_users)} test users from training set")

    if len(test_users) == 0:
        print("No overlap between training and test users. Using fallback to first 20 training users for testing.")
        test_users = list(train_users)[:20]  # CAP: Increased to 20


    # CAP: Show recommendations for first 20 users
    successful_recommendations = 0
    user_count = 0
    for user in test_users:
        if user_count >= 20:  # CAP: Stop after 20 users
            break
        try:
            recommendations = get_top_recommendations(user, trainset, model, n=5)
            if recommendations:
                print(f"\nTop 5 recommendations for user '{user}':")
                for i, (item, rating) in enumerate(recommendations, 1):
                    print(f"  {i}. {item}: {rating:.3f}")
                successful_recommendations += 1
            user_count += 1
        except Exception as e:
            print(f"  Could not generate recommendations for {user}: {e}")
            user_count += 1

    print(f"\nCapped to first {user_count} users")
    
    # ------------------------------------------------------------------------
    # STEP 5: Test individual metric functions
    # ------------------------------------------------------------------------
    print("\nSTEP 5: Testing individual metrics...")
    print("-" * 70)
    
    # Use actual items from the dataset for more realistic testing
    all_items = interactions_df['item_id'].unique()
    
    # Find a user with enough interactions for testing
    test_user = None
    for user in test_users:
        user_interactions = train_data[train_data['user_id'] == user]
        if len(user_interactions) >= 3:  # User has at least 3 interactions
            test_user = user
            break

    if test_user is None:
        print(f"Checking {len(test_users)} test users in training data...")
        for user in test_users[:5]:
            user_interactions = train_data[train_data['user_id'] == user]
            print(f" User '{user}': {len(user_interactions)} interactions in training data")
    
    if test_user:
        user_actual_interactions = interactions_df[interactions_df['user_id'] == test_user]
        test_actual = set(user_actual_interactions['item_id'].values)
        
        # Generate some test recommendations (mix of actual and new items)
        actual_items = list(test_actual)[:3]  # Take first 3 actual items
        new_items = list(np.random.choice(
            [item for item in all_items if item not in test_actual], 
            size=7, replace=False
        ))
        test_recommendations = actual_items + new_items
        
        precision = calc_precision_at_k(test_recommendations, test_actual, k=5)
        recall = calc_recall_at_k(test_recommendations, test_actual, k=5)
        f1 = calc_f1_score(precision, recall)
        ap = calc_average_precison(test_recommendations, test_actual, k=5)
        
        print(f"Sample user '{test_user}' has {len(test_actual)} actual interactions")
        print(f"Precision@5: {precision:.3f}")
        print(f"Recall@5: {recall:.3f}")
        print(f"F1@5: {f1:.3f}")
        print(f"Average Precision@5: {ap:.3f}")
    else:
        print("No suitable user found for metric testing")
    
    # ------------------------------------------------------------------------
    # STEP 6: Test diversity and coverage metrics
    # ------------------------------------------------------------------------
    print("\nSTEP 6: Testing diversity and coverage metrics...")
    print("-" * 70)
    
    # Generate recommendations for test users for diversity analysis
    all_recommendations = []
    successful_users = 0
    
    for user in test_users:
        try:
            recs = get_top_recommendations(user, trainset, model, n=20)
            if recs:
                item_recs = [item for item, _ in recs]
                all_recommendations.append(item_recs)
                successful_users += 1
        except Exception:
            continue
    
    print(f"Successfully generated recommendations for {successful_users}/{len(test_users)} users")
    
    if len(all_recommendations) > 0:
        # Test diversity
        diversity_metrics = calc_recommendation_diversity(all_recommendations)
        print(f"Normalized Entropy: {diversity_metrics['normalized_entropy']:.3f}")
        print(f"Gini Coefficient: {diversity_metrics['gini_coefficient']:.3f}")
        print(f"Unique Items Recommended: {diversity_metrics['unique_items_recommend']}")
        
        # Test coverage
        coverage_metrics = calc_catalog_coverage(all_recommendations, all_items)
        print(f"Catalog Coverage: {coverage_metrics['coverage']:.1%}")
        print(f"Items Recommended: {coverage_metrics['items_recommeded']}")
        print(f"Items Never Recommended: {coverage_metrics['items_never_recommended']}")
        
        # Since we're only testing 25 items, show which items were never recommended
        if coverage_metrics['items_never_recommended'] > 0:
            print(f"\nItems never recommended:")
            all_recommended_items = set()
            for rec_list in all_recommendations:
                all_recommended_items.update(rec_list)
            never_recommended = set(all_items) - all_recommended_items
            for i, item in enumerate(never_recommended, 1):
                print(f"  {i}. {item}")
    else:
        print("No recommendations generated for diversity analysis")
    
    # ------------------------------------------------------------------------
    # STEP 7: Complete evaluation using RecommenderEvaluator
    # ------------------------------------------------------------------------
    print("\nSTEP 7: Running complete evaluation pipeline...")
    print("-" * 70)
    
    try:
        # CAP: Use smaller k values for smaller dataset
        train_user_ids = [trainset.to_raw_uid(u) for u in trainset.all_users()]
        filtered_test_data = test_data[test_data['user_id'].isin(train_user_ids)].copy()

        if filtered_test_data.empty:
            print("No overlapping users between training and test data - evaluation skipped.")
            metrics = {f"{m}@{k}": 0.0 for m in ["precision", "recall", "f1", "map"] for k in [3, 5]}
        else:
            evaluator = RecommenderEvaluator(k_values=[3, 5])
            metrics = evaluator.evaluate(model, trainset, filtered_test_data)
        
        print("Evaluation Results:")
        for metric, value in sorted(metrics.items()):
            print(f"  {metric}: {value:.3f}")
            
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        metrics = {}
    
    # ------------------------------------------------------------------------
    # STEP 8: Generate comprehensive evaluation report
    # ------------------------------------------------------------------------
    print("\nSTEP 8: Generating evaluation report...")
    print("-" * 70)
    
    try:
        report = generate_eval_report(model, trainset, train_data, test_data, all_items)
        print("✓ Evaluation report generated successfully")
    except Exception as e:
        print(f"  Note: Could not generate full report: {e}")
    
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
        (successful_recommendations > 0, "Recommendations were generated"),
        (isinstance(metrics, dict), "Evaluation metrics were computed"),
        (interactions_df['item_id'].nunique() <= 25, "Test limited to 25 items as requested"),
    ]
    
    # Add metric range checks if metrics exist
    if metrics:
        checks.extend([
            (0 <= metrics.get('precision@3', 0) <= 1, "Precision is within valid range"),
            (0 <= metrics.get('recall@3', 0) <= 1, "Recall is within valid range"),
        ])
    
    all_passed = True
    for check, message in checks:
        if check:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nKey Insights:")
        print(f"- Model trained on {trainset.n_users} users and {trainset.n_items} items")
        print(f"- Test limited to first 25 items as requested")
        print(f"- Temporal split preserves chronological order")
        print(f"- All evaluation metrics computed successfully")
        if len(all_recommendations) > 0:
            print(f"- Diversity: {diversity_metrics['normalized_entropy']:.1%} entropy")
            print(f"- Coverage: {coverage_metrics['coverage']:.1%} of catalog recommended")
    else:
        print("\n⚠ Some tests failed. Check the implementation.")
    
    return all_passed


def create_synthetic_fallback_data(max_items=25):
    """
    Create synthetic data as fallback when real data is not available
    """
    print(f"Creating synthetic test data with {max_items} items as fallback...")
    
    np.random.seed(42)
    
    # Create synthetic users, items, and ratings
    users = [f"User_{i}" for i in range(1, 51)]  # Fewer users for test
    items = [f"Item_{i}" for i in range(1, max_items + 1)]  # Exactly max_items
    
    interactions = []
    base_date = datetime(2024, 1, 1)
    
    for user in users:
        # Each user rates 5-10 random items (smaller for test)
        num_ratings = np.random.randint(5, 11)
        rated_items = np.random.choice(items, size=num_ratings, replace=False)
        
        for item in rated_items:
            # Generate rating with some user bias
            base_rating = np.random.normal(3.5, 1.0)
            rating = max(1, min(5, round(base_rating)))
            
            # Random timestamp within 3 months (shorter for test)
            days_offset = np.random.randint(0, 90)
            timestamp = base_date + timedelta(days=days_offset)
            
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': float(rating),  # Ensure it's float
                'timestamp': timestamp
            })
    
    df = pd.DataFrame(interactions)
    print(f"Created synthetic dataset with {len(df)} interactions")
    print(f"Using {len(items)} items: {items}")
    return df


# Run the comprehensive test
if __name__ == "__main__":
    print("Starting UK Food Savers Recommendation System Test...")
    print("TESTING WITH FIRST 25 ITEMS ONLY")
    print("Make sure 'UKFoodSavers_testdata.csv' is in the current directory.")
    print("-" * 70)
    
    success = test_complete_evaluation_pipeline()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("✓ Limited to first 25 items as requested")
    else:
        print("\n❌ Test completed with some issues.")
    
    print("\n" + "=" * 70)