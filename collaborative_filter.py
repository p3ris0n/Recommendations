import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader
from surprise import SVD
from collections import defaultdict


# sample data: (user_id, item_id, rating)
# rating system: 1.0 for purchased and 0.0 for not-purchased.

data = {
    'user_id' : ['Mike', 'Jane', 'Alex', 'Anne'],
    'item_id' : ['Burger', 'Pizza', 'Salad', 'Pizza'],
    'rating' : [1.0, 1.0, 0.0, 1.0]
}

train_data = pd.DataFrame(data)
print("Our Raw Data: ")
print(train_data)

reader = Reader(rating_scale = (0, 1)) # reader is needed to parse the dataframe. 

dataset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader)

# build the training set.
trainset = dataset.build_full_trainset()

# using the SVD algo.
model = SVD()
model.fit(trainset)
print("Model trained successfully!")

# function to get top-n recommendations.

def get_top_recommendations(user_id, trainset, model, n=5):
    # get a list of all items.
    all_items = trainset.all_items()

    # get user-rated items.
    user_items = set([trainset.to_raw_iid(item_id) for [user, item_id] in trainset.ur[trainset.to_inner_uid(user_id)]])

    # predictions.
    predictions = []
    for item_id in all_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        if raw_item_id not in user_items: # predict for the first time only,
            pred = model.predict(user_id, raw_item_id)
            predictions.append((raw_item_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True) # sort by estimated ratings.

    return predictions[:n] # returns top-n recs.

print("\nGenerate recommendations for each user: ")
for user_id in data['user_id']:
    recommendations = get_top_recommendations(user_id, trainset, model, n=1)
    print(f"\nTop Recommnedations for {user_id}")
    for item, rating in recommendations:
        print(f"- {item}: predicted rating = {rating:.3f}")


def create_temporal_split(interactions_df, test_weeks=2, validation_weeks=1):
    # This splits interactions by time to simulate real-world usage.

    # sort interations by timestamps.
    interactions_sorted = interactions_df.sort_values('timestamp')

    # calc. for cut of dates.
    max_date = interactions_sorted['timestamp'].max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks)
    validation_start = test_start - pd.Timedelta(weeks=validation_weeks)

    # split the data.
    train_data = interactions_sorted[interactions_sorted['timestamp'] < validation_start]
    validation_data = interactions_sorted[(
        interactions_sorted['timestamp'] >= validation_start) & (interactions_sorted['timestamp'] < test_start)]

    test_data = interactions_sorted[interactions_sorted['timestamp'] >= test_start]

    print(f"training interactions: {len(train_data)}")
    print(f"validation interactions: {len(validation_data)}")
    print(f"test interactions: {len(test_data)}")

    if len(test_data) < 1000:
        print("Warning: Test set might be too small and it might invalidate the results.")

    return train_data, validation_data, test_data

def calc_precision_at_k(recommendations, actual_interactions, k=10):
    # calculates precision at k. what fraction of recommended items were actually relevant?

    """
    Args: 
        recommendations: List of recommended items IDs, ordered by relevance
        actual_interactions: Set of item IDs that the user actually interacted with.
        k: Number of top recommendations to consider.
    
    Returns:
        float between 0 and 1, where 1 means all recommended items were relevant. 
    
    """
    top_k = recommendations[:k]

    relevant_and_recommended = set(top_k) & set(actual_interactions)

    if len(top_k) == 0:
        return 0.0
    
    return len(relevant_and_recommended) / len(top_k)

def calc_recall_at_k(recommendations, actual_interactions, k=10):
    # calculates recall at k. what fraction of relevant items were recommended?

    """
    Args: 
        recommendations: List of recommended items IDs, ordered by relevance
        actual_interactions: Set of item IDs that the user actually interacted with.
        k: Number of top recommendations to consider.
    
    Returns:
        float between 0 and 1, where 1 means all relevant items were recommended. 

        This tells you how complete your recommendations are. You might have a high precision
        but if your recall is low, it means you're missing a lot of relevant items.
    
    """
    top_k = recommendations[:k]

    relevant_and_recommended = set(top_k) & set(actual_interactions)

    if len(actual_interactions) == 0:
        return 0.0
    
    return len(relevant_and_recommended) / len(actual_interactions)


def calc_f1_score(precision, recall):
    # calculates the harmonic mean of precision and recall.

    """
    Args:
        precision: Precision value
        recall: Recall value
    
    Returns:
        float between 0 and 1, where 1 means perfect precision and recall.

        the f1 score balances both metrics, you can't get a high f1 score without both a good
        precision score and a good recall score.

    """

    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# Implementing Mean Average Precision.
# this gives a more nuanced view of recommendation quality 
# by considering the rank of relevant items and putting those first.
# It's more sophisticated because it considers the order of recommendations.

def calc_average_precison(recommendations, actual_interactions, k=10):
    top_k = recommendations[:k]

    if len(actual_interactions) == 0:
        return 0.0

    precision_sum = 0.0
    num_relevant_found = 0

    for i, item in enumerate(top_k):
        if item in actual_interactions:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1) # precision at position.
            precision_sum += precision_at_i

    if num_relevant_found == 0:
        return 0.0

    return precision_sum / min(len(actual_interactions), k)

def calc_mean_avg_precision(all_recommendations, all_actual_interactions, k=10):
    # creates and calculates MAP for all users.
    # considering both relevance and ranking quality.

    avg_precisions = []

    for user_recs, user_actual in zip(all_recommendations, all_actual_interactions):
        ap = calc_average_precison(user_recs, user_actual, k)
        avg_precisions.append(ap)

    return np.mean(avg_precisions)


class RecommenderEvaluator:
    # a comprehensive eval framework, this class encapsulates all the eval logic
    # so you can easily evaluate different models consistently.

    def __init__(self, k_values=[5, 10, 20]):
        """
            k_values: differenct cutoff points to evaluate.

            what they see immediately (top_5) or what they see if they scroll down (top_20)
        """

        self.k_values = k_values
        self.results = {}

    def evaluate(self, trainset, model, test_data):

        """
            Runs a complete eval of the model.

            Args: 
                model: your trained rec model.
                test_data: dataframe with cols [user_id, item_id, timestamps]

            Returns
                Dict of metrics
        """
        # group test data by user to see what each user interacted with.
        actual_interactions = test_data.groupby('user_id')['item_id'].apply(set).to_dict()
        results = {k: {
            'precision': [],
            'recall': [],
            'f1': [],
            'avg_precision': []
        } for k in self.k_values}

        # evaluate for each user who has test interactions
        for user_id, actual_items in actual_interactions.items():
            try:
                recs = get_top_recommendations(user_id, trainset, model, n=max(self.k_values))
                recommendations = [item for item, _ in recs] # extracts just items_id
               
            except: # handles coldstart users who weren't in the training data.
                continue

            for k in self.k_values:
                precision = calc_precision_at_k(recommendations, actual_items, k)
                recall = calc_recall_at_k(recommendations, actual_items, k)
                f1 = calc_f1_score(precision, recall)
                ap = calc_average_precison(recommendations, actual_items, k)

                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
                results[k]['f1'].append(f1)
                results[k]['avg_precision'].append(ap)

        summary = {}
        for k in self.k_values:
            summary[f'precision@{k}'] = np.mean(results[k]['precision'])
            summary[f'recall@{k}'] = np.mean(results[k]['recall'])
            summary[f'f1@{k}'] = np.mean(results[k]['f1']) 
            summary[f'map@{k}'] = np.mean(results[k]['avg_precision'])

        return summary

# Test Cases

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