import sys
import traceback

def run_tests():
    try:
        from utils.preprocessing import load_data, get_data_summary, clean_and_preprocess
        from utils.clustering import find_optimal_k, train_kmeans, assign_personas, generate_insights
        from utils.visualization import (plot_distribution, plot_correlation_heatmap, 
                                         plot_pairplot, plot_scatter, plot_elbow_curve, 
                                         plot_silhouette_scores, plot_3d_clusters, 
                                         plot_cluster_radar, plot_cluster_distribution)
        from utils.prediction import load_models, predict_customer_segment
        
        print("1. Loading Data...")
        df = load_data()
        if df is None:
            print("FAILED: Data not loaded.")
            return
            
        print("2. Summary...")
        summary = get_data_summary(df)
        
        print("3. Preprocessing...")
        df_clean, X_scaled, features = clean_and_preprocess(df)
        
        print("4. Evaluating K...")
        metrics_df = find_optimal_k(X_scaled, max_k=10)
        
        print("5. Training KMeans...")
        optimal_k = 5
        kmeans = train_kmeans(X_scaled, optimal_k)
        
        print("6. Personas...")
        df_clean['Cluster'] = kmeans.labels_
        df_clean, personas_map = assign_personas(df_clean, 'Cluster')
        insights = generate_insights(personas_map)
        
        print("7. Visualizations (building figures)...")
        plot_distribution(df, 'Age', 'Age Dist')
        plot_correlation_heatmap(df)
        plot_pairplot(df)
        plot_scatter(df, 'Age', 'Spending Score (1-100)')
        plot_elbow_curve(metrics_df)
        plot_silhouette_scores(metrics_df)
        plot_3d_clusters(df_clean)
        plot_cluster_radar(df_clean)
        plot_cluster_distribution(df_clean)
        
        print("8. Prediction...")
        scaler, loaded_kmeans = load_models()
        res = predict_customer_segment(30, 50, 50, scaler, loaded_kmeans, personas_map)
        
        print("ALL TESTS PASSED SUCCESSFULLY.")
        
    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
