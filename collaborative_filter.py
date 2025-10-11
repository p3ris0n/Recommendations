import kagglehub
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

def calc_recommendation_diversity(all_recommendations):
    
    """
        Maximum enthropy ensures the recommendations are evenly distrubuted across all itmes.
    """
    # counts how many times each items was recommended.
    item_counts = {}
    total_recommendations = 0

    for recommendations in all_recommendations:
        for item in recommendations:
            item_counts[item] = item_counts.get(item, 0) + 1
            total_recommendations += 1

    entropy = 0
    for count in item_counts.values():
        probability = count / total_recommendations
        if probability > 0:
            entropy -= probability * np.log2(probability)

    max_entropy = np.log2(len(item_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'unique_items_recommend': len(item_counts),
        'gini_coefficient': calc_gini(list(item_counts.values()))
    }

def calc_gini(counts):
    """
    Calculate Gini coefficient to measure inequality in recommendations.
    
    Gini = 0 means perfect equality (all items recommended equally)
    Gini = 1 means perfect inequality (all recommendations for one item)
    
    For food waste, you want a lower Gini coefficient.
    """
    counts = np.array(sorted(counts))
    n = len(counts)
    index = np.array(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n - 1) / n

def calc_catalog_coverage(all_recommendations, total_available_itmes):
    recommended_items = set()
    for recommendations in all_recommendations:
        recommended_items.update(recommendations)

    coverage = len(recommended_items)/len(total_available_itmes)

    return {
        'coverage': coverage,
        'items_recommeded': len(recommended_items),
        'items_never_recommended': len(total_available_itmes) - len(recommended_items)
    }