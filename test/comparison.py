from recordlinkage.datasets import load_febrl4
import pandas as pd
import re
import numpy as np
import recordlinkage


def comparison(no_of_queries):
    no_of_queries = no_of_queries
    results_private=pd.read_csv('out/results_' + str(no_of_queries) + '.csv')
    results_private.rename(columns={'key':'state','average': 'Average_Age'},inplace=True)
    print(results_private)
    dfA, dfB, true_links = load_febrl4(return_links=True)
    
    # Define number of queries (adjust as needed)
      # Change this value as required
    
    # Dataset 1: First `no_of_queries` records
    #dataset1 = dfA.iloc[:no_of_queries].copy()
    dname = 'df_names_10k_lsh200-50-100'
    df1 = pd.read_pickle("dataset/"+ dname + ".pkl")
    dataset1 = df1.iloc[:no_of_queries][['ID','Full Name']]
    
    
    payload_var='age'
    cross_tab_var='state'
    dname = 'df_fuzzy_names_10k_lsh200-50-100'
    df2 = pd.read_pickle("dataset/"+ dname + ".pkl")
    print(df2.columns)
    dataset2= df2[['ID','Fuzzy Full Name',payload_var,cross_tab_var]]
    
    # Indexer to generate candidate pairs
    indexer = recordlinkage.Index()
    indexer.full()  # Compare all pairs
    candidate_pairs = indexer.index(dataset1, dataset2)
    
    # Define record linkage using Levenshtein similarity
    compare = recordlinkage.Compare()
    compare.string('Full Name', 'Fuzzy Full Name', method='levenshtein', threshold=0.8, label='name_similarity')
    
    # Compute similarity scores
    features = compare.compute(candidate_pairs, dataset1, dataset2)
    
    # Get matched records
    matches = features[features['name_similarity'] == 1].reset_index()
    
    matches.drop_duplicates(subset=['level_0'],keep='first',inplace=True)
    # Merge matches with dataset2 to get state and age
    matched_dataset2 = matches.merge(dataset2, left_on='level_1', right_index=True)
    
    # Compute average age per state (from dataset2)
    #average_age_per_state = matched_dataset2.groupby('state')['Year of Birth'].mean()
    
    # Display the result
    #print(average_age_per_state)
    
    
    results_rl= matched_dataset2.groupby(cross_tab_var).agg(
        Average_Age=(payload_var, 'mean'),
        Count=(cross_tab_var, 'count')
    )

    #Comparison between direct computation and private computation
    
    results_direct= df2[df2['ID'].isin(df1[:no_of_queries]['ID'])][[payload_var, cross_tab_var]] \
        .groupby(cross_tab_var) \
        .agg(**{
            'Average_Age': (payload_var, 'mean'),
            'Count': (payload_var, 'count')
        })

    # TRUE POSITIVES (TP): Correctly matched IDs
    matched_dataset2 = matched_dataset2.merge(dataset1, left_on='level_0', right_index=True, suffixes=('_2', '_1'))
    true_positives = (matched_dataset2['ID_1'] == matched_dataset2['ID_2']).sum()
    
    # FALSE POSITIVES (FP): Incorrectly matched records (wrong ID pairs)
    false_positives = len(matches) - true_positives
    
    # TRUE LINKS: Expected true matches should be at most `no_of_queries`
    max_true_links = no_of_queries  
    
    # FALSE NEGATIVES (FN): Records that should have matched but were not found
    found_links = set(zip(matched_dataset2['ID_1'], matched_dataset2['ID_2']))  # Detected links
    false_negatives = max_true_links - true_positives  # Missed matches
    
    # TOTAL POSSIBLE COMPARISONS (N1 * N2)
    total_comparisons = len(dataset1) * len(dataset2)
    
    # TRUE NEGATIVES (TN): Pairs correctly identified as non-matches
    true_negatives = total_comparisons - (true_positives + false_positives + false_negatives)
    
    # Precision, Recall, F1-score, and Accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / max_true_links if max_true_links > 0 else 0  # Normalized recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / total_comparisons if total_comparisons > 0 else 0
    
    # Print Performance Metrics
    print("\nRecord Linkage Performance Metrics:")
    print(f"True Positives (TP): {true_positives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"True Negatives (TN): {true_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    results_merged=results_rl.merge(results_private,left_on='state',right_on='state',suffixes=('_rl','_private'))
    results_merged=results_merged.merge(results_direct,left_on='state',right_on='state',suffixes=('_','_actual'))
    
    df=results_merged
    # Calculating the differences
    rmse_rl = np.sqrt(((df['Average_Age_rl'] - df['Average_Age']) ** 2).mean())
    rmse_private = np.sqrt(((df['Average_Age_private'] - df['Average_Age']) ** 2).mean())
    #df["Diff_Age_rl"] = df["Average_Age_rl"] - df["Average_Age"]
    #df["Diff_Age_private"] = df["Average_Age_private"] - df["Average_Age"]
    #df["Diff_Count_rl"] = df["Count_rl"] - df["Count"]
    #df["Diff_Count_private"] = df["Count_private"] - df["Count"]
    
    # Computing the mean differences
    #mean_diff_age_rl = df["Diff_Age_rl"].mean()
    #mean_diff_age_private = df["Diff_Age_private"].mean()
    #mean_diff_count_rl = df["Diff_Count_rl"].mean()
    #mean_diff_count_private = df["Diff_Count_private"].mean()
    
    # Printing the results
    #print("Mean Difference in Age (RL vs Ground Truth):", mean_diff_age_rl)
    #print("Mean Difference in Age (Private vs Ground Truth):", mean_diff_age_private)
    print("Root Mean Square Difference in Age (RL vs Ground Truth):",rmse_rl)
    print("Root Mean Square Difference in Age (Private vs Ground Truth):", rmse_private)

    return results_merged


df=comparison(5000)
print(df)