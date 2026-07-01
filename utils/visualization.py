import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ----------------- EDA Visualizations ----------------- #

def plot_distribution(df, column, title, color_sequence=None):
    fig = px.histogram(df, x=column, title=title, 
                       marginal="box", color_discrete_sequence=color_sequence,
                       template="plotly_dark")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                    color_continuous_scale='RdBu_r', 
                    title="Correlation Heatmap", template="plotly_dark")
    return fig

def plot_pairplot(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'CustomerID' in numeric_cols:
        numeric_cols = numeric_cols.drop('CustomerID')
        
    fig = px.scatter_matrix(df, dimensions=numeric_cols, 
                            color="Gender" if "Gender" in df.columns else None,
                            title="Pair Plot of Numerical Features",
                            template="plotly_dark")
    fig.update_traces(diagonal_visible=False)
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, title="Scatter Plot"):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                     title=title, template="plotly_dark",
                     hover_data=['Age', 'Gender'] if 'Age' in df.columns else None)
    return fig

# ----------------- Clustering Visualizations ----------------- #

def plot_elbow_curve(metrics_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics_df['k'], y=metrics_df['inertia'], 
                             mode='lines+markers', name='Inertia',
                             line=dict(color='cyan', width=2)))
    fig.update_layout(title='Elbow Method For Optimal k',
                      xaxis_title='Number of Clusters (k)',
                      yaxis_title='Inertia',
                      template="plotly_dark")
    return fig

def plot_silhouette_scores(metrics_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_df['k'], y=metrics_df['silhouette'], 
                         marker_color='orange'))
    fig.update_layout(title='Silhouette Scores per k',
                      xaxis_title='Number of Clusters (k)',
                      yaxis_title='Silhouette Score',
                      template="plotly_dark")
    return fig

def plot_3d_clusters(df, cluster_col='Cluster', title="3D Cluster View"):
    fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                        color=cluster_col, symbol=cluster_col,
                        title=title, template="plotly_dark",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        opacity=0.8)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_cluster_radar(df, cluster_col='Cluster'):
    # Mean of features per cluster
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    cluster_means = df.groupby(cluster_col)[features].mean().reset_index()
    
    # Scale means to 0-1 range just for radar chart visualization
    for feature in features:
        max_val = df[feature].max()
        cluster_means[feature] = cluster_means[feature] / max_val
    
    fig = go.Figure()
    for index, row in cluster_means.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)']],
            theta=features,
            fill='toself',
            name=f"Cluster {int(row[cluster_col])} ({row.get('Persona', '')})" if 'Persona' in row else f"Cluster {int(row[cluster_col])}"
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Cluster Profiles (Normalized)",
        template="plotly_dark"
    )
    return fig

def plot_cluster_distribution(df, cluster_col='Cluster'):
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = [cluster_col, 'Count']
    # If Persona is mapped, we can use it
    if 'Persona' in df.columns:
        persona_map = df.drop_duplicates(subset=[cluster_col, 'Persona'])[[cluster_col, 'Persona']]
        counts = counts.merge(persona_map, on=cluster_col)
        counts['Label'] = counts['Persona'] + " (C" + counts[cluster_col].astype(str) + ")"
    else:
        counts['Label'] = "Cluster " + counts[cluster_col].astype(str)
        
    fig = px.pie(counts, values='Count', names='Label', 
                 title="Cluster Distribution", template="plotly_dark",
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig
