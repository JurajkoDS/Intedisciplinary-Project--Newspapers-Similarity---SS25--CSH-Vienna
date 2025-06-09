import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import regex as re
import networkx as nx
import random
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import plotly.graph_objects as go
from PIL import Image

from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity

def get_matrix(month):

    retweeter_to_index = {retweeter: i for i, retweeter in enumerate(month['retweeter_id'].unique())}
    author_to_index = {author: j for j, author in enumerate(month['author_id'].unique())}

    #print(author_to_index)
    #print(len(author_to_index))

    # Map retweeters and authors to indices
    rows = month['retweeter_id'].map(retweeter_to_index)
    cols = month['author_id'].map(author_to_index)

    #print(cols)

    # Create sparse biadjacency matrix
    return pd.DataFrame({'Retweeters': rows, 'Authors': cols})


def get_similarities(month_matrix: pd.DataFrame, metric: str):
    if metric == 'jaccard':
        # Pivot the DataFrame to create a biadjacency matrix
        biadjacency_matrix = month_matrix.pivot_table(index='Retweeters', columns='Authors', aggfunc='size', fill_value=0).astype(int)

        # Extract rows (X) and columns (Y) of the biadjacency matrix
        X = biadjacency_matrix.values.T  # Rows of the biadjacency matrix
        Y = X

        # Compute Jaccard similarities
        similarities_array = 1 - distance.cdist(X, Y, metric='jaccard')
        return pd.DataFrame(similarities_array, index=biadjacency_matrix.columns, columns=biadjacency_matrix.columns)
    
    elif metric == 'cosine':
        # Pivot the DataFrame to create a biadjacency matrix
        biadjacency_matrix = month_matrix.pivot_table(index='Retweeters', columns='Authors', aggfunc='size', fill_value=0).astype(int)

        # Extract rows (X) and columns (Y) of the biadjacency matrix
        X = biadjacency_matrix.values.T  # Columns of the biadjacency matrix
        Y = X

        similarities_array = 1 - distance.cdist(X, Y, metric='cosine')
        return pd.DataFrame(similarities_array, index=biadjacency_matrix.columns, columns=biadjacency_matrix.columns)
    

def top_bottom_similarities(month_name:str):
    
    '''
    This function outputs the top and bottom 10 similarities for a specific month.
    '''

    author_to_index = {author: j for j, author in enumerate(month_data[month_name]['author_id'].unique())}

    ## Create a mapping from author_id to author_name
    author_id_to_name = month_data[month_name].set_index('author_id')['author_name'].to_dict()

    #  Map author IDs to names
    author_names = [author_id_to_name[author] for author in author_to_index.keys()]

    # Create a DataFrame for the similarities matrix
    similarities_df = similarities[month_name].copy()
    similarities_df.columns = author_names
    similarities_df.index = author_names

    # Flatten the DataFrame into a long format
    similarities_long = similarities_df.stack().reset_index()
    similarities_long.columns = ['Author 1', 'Author 2', 'Similarity']

    # Drop NaN (diagonal entries) and duplicates (since matrix is symmetric)
    similarities_long = similarities_long.dropna()
    similarities_long = similarities_long[similarities_long['Author 1'] != similarities_long['Author 2']]

    # Remove duplicate pairs (A-B is the same as B-A in similarity matrices)
    similarities_long['Sorted_Pair'] = similarities_long.apply(lambda x: tuple(sorted([x['Author 1'], x['Author 2']])), axis=1)
    similarities_long = similarities_long.drop_duplicates('Sorted_Pair').drop(columns=['Sorted_Pair'])

    # Get top 10 highest and lowest similarities
    top10_highest = similarities_long.nlargest(10, 'Similarity')
    top10_lowest = similarities_long.nsmallest(10, 'Similarity')

    # Save the results as CSV files in the year folder
    top10_highest.to_csv(os.path.join(year_folder, f"top10_highest_similarities_{month_name}_{year}.csv"), index=False)
    top10_lowest.to_csv(os.path.join(year_folder, f"top10_lowest_similarities_{month_name}_{year}.csv"), index=False)


def plot_pca_for_months_interactive(pca_df, month_names, save=False):
        """
        Plots interactive PCA results for each month with node names shown on hover
        and a toggle to display node index labels.

        Parameters:
        - pca_df (pd.DataFrame): DataFrame containing PCA results with index containing month names.
        - month_names (list): List of month names to filter and plot.

        Returns:
        - None: Displays the interactive plots.
        """
        x_range = [-0.5, 2]
        y_range = [-0.5, 2]
        x_ticks = np.arange(x_range[0], x_range[1] + 0.25, 0.25)
        y_ticks = np.arange(y_range[0], y_range[1] + 0.25, 0.25)

        for month in month_names:
            month_data = pca_df[pca_df.index.str.contains(f'_{month}', case=False)].sort_index()
            month_data = month_data.copy()
            month_data['node'] = month_data.index.str.split('_').str[0]
            month_data['index'] = month_data.index

            # Create base scatter plot without text labels
            scatter = go.Scatter(
                x=month_data['PC1'],
                y=month_data['PC2'],
                mode='markers+text',
                text=month_data['index'],
                textposition='top center',
                marker=dict(size=8, opacity=0.7),
                hoverinfo='text',
                hovertext=month_data['index'],
                name=''
            )

            fig = go.Figure(data=[scatter])

            # Initially hide the text
            fig.update_traces(textfont=dict(size=10), text=None)

            # Add toggle button
            fig.update_layout(
                title=f"PCA Plot for {month}",
                width=800,
                height=600,
                xaxis=dict(
                    title='PC1',
                    showgrid=True,
                    range=x_range,
                    tickvals=x_ticks
                ),
                yaxis=dict(
                    title='PC2',
                    showgrid=True,
                    range=y_range,
                    tickvals=y_ticks
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.7,
                        y=1.15,
                        buttons=list([
                            dict(label="Show Labels",
                                method="restyle",
                                args=[{"text": [month_data['index']]}]),
                            dict(label="Hide Labels",
                                method="restyle",
                                args=[{"text": [None]}])
                        ]),
                        showactive=True
                    )
                ],
                showlegend=False
            )

            #fig.show()

            if save:
                # Save the figure as an image in the 'images' folder
                month_number = month_names.index(month) + 1  # 1-based indexing
                fig.write_image(os.path.join(year_folder, f"{month_number}_PCA_Plot_{month}_{year}.png"))

def calculate_monthly_velocities_cosine(df, month_names):
        """
        Calculates the velocities (cosine distances) between consecutive months.

        Parameters:
        - df (pd.DataFrame): DataFrame containing results with index containing month names.
        - month_names (list): List of month names in chronological order.

        Returns:
        - velocities (dict): Dictionary where keys are month pairs (e.g., "January-February") and values are DataFrames
                            containing velocities for each node.
        """
        velocities = {}

        for i in range(len(month_names) - 1):
            # Get the current and next months
            current_month = month_names[i]
            next_month = month_names[i + 1]

            # Filter PCA data for the two months
            current_data = df[df.index.str.contains(f'_{current_month}', case=False)].sort_index()
            next_data = df[df.index.str.contains(f'_{next_month}', case=False)].sort_index()

            # Extract base node names (before '_')
            current_data.index = current_data.index.str.split('_').str[0]
            next_data.index = next_data.index.str.split('_').str[0]

            # Get union of indices to ensure all nodes are included
            all_indices = current_data.index.union(next_data.index)

            # Reindex both DataFrames to include all nodes, filling missing values with NaN
            current_data = current_data.reindex(all_indices)
            next_data = next_data.reindex(all_indices)

            # Calculate cosine distance (velocity) for each node
            distances = []
            for node in all_indices:
                vec1 = current_data.loc[node].fillna(0)
                vec2 = next_data.loc[node].fillna(0)

                # Check for zero vectors
                if np.all(vec1 == 0) or np.all(vec2 == 0): # missing in one month or the other
                    distances.append(np.nan)  # assign maximum distance of 1 or declare "missing"?

                #elif np.all(vec1 == 0) and np.all(vec2 == 0): # missing in both months - actually doesn't occur, because we only build on the union of the one month and the month following, not on all nodes
                    #distances.append(np.nan)
                else:
                    distances.append(cosine(vec1, vec2))

            # Store the velocities as a DataFrame in a dictionary velocities
            velocities[f"{current_month}-{next_month}"] = pd.DataFrame({
                'Node': all_indices,
                'Velocity': distances
            }).set_index('Node')

        return velocities


# Plotting the KDE overlay for all month pairs
def plot_velocity_kde_overlay(velocities_df, common_norm: bool):
    """
    Plots all monthly KDE lines from velocities_df on a single figure for comparison.
    """
    plt.figure(figsize=(10, 6))
    for column in velocities_df.columns:
        data = velocities_df[column].dropna()
        sns.kdeplot(data, label=column, linewidth=2, clip=(0, None), common_norm=common_norm) # clip parameter, to prevent distances to plotting (smoothing the distances) to lower than 0
    
    if common_norm:
        plt.title(f"{year}, Velocity KDE Overlay for All Month Pairs (Normalized)")
    else:
        plt.title("Velocity KDE Overlay for All Month Pairs (Not Normalized)")
        
    plt.xlabel("Velocity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(year_folder, f"velocity_histograms_{year}.png"))
    #plt.show()



def concatenate_yearly_graphs(years, folder='.', output_filename='combined_2018_2022.png'):
    images = []
    for year in years:
        img_path = os.path.join(folder, str(year), f"combined_{year}.png")
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            print(f"Warning: {img_path} not found, skipping.")
    if not images:
        print("No images found to concatenate.")
        return
    # Resize all images to the same height
    min_height = min(img.height for img in images)
    images = [img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS) for img in images]
    total_width = sum(img.width for img in images)
    combined_img = Image.new('RGB', (total_width, min_height))
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
    combined_img.save(output_filename)
    print(f"Saved concatenated image as {output_filename}")


def plot_combined_years(data_per_year, years, column_order):
    """
    data_per_year: dict of {year: (means_arr, stds_arr, consecutive_modularity_averages, consecutive_month_pairs)}
    years: list of years to plot
    column_order: list of all month-pair labels in order
    """
    # Concatenate data
    all_means = []
    all_stds = []
    all_modularities = []
    all_month_pairs = []
    for year in years:
        means_arr, stds_arr, modularity_avgs, month_pairs = data_per_year[year]
        all_means.extend(means_arr)
        all_stds.extend(stds_arr)
        all_modularities.extend(modularity_avgs)
        all_month_pairs.extend([f"{year}-{mp}" for mp in month_pairs])

    fig, ax1 = plt.subplots(figsize=(24, 6))
    ax1.plot(all_month_pairs, all_modularities, marker='o', linestyle='-', color='purple', label='Avg. Modularity')
    ax1.set_xlabel('Year-Month Pair')
    ax1.set_ylabel('Average Modularity', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_xticks(range(len(all_month_pairs)))
    ax1.set_xticklabels(all_month_pairs, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(all_month_pairs, all_means, marker='o', linestyle='--', color='g', label='Mean Velocity')
    ax2.fill_between(range(len(all_month_pairs)), np.array(all_means) - np.array(all_stds), np.array(all_means) + np.array(all_stds), color='green', alpha=0.15, label='Velocity Mean ± Std. Dev.')
    ax2.set_ylabel('Velocity', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1.02, 0.5), borderaxespad=6)

    plt.title('Combined Years: Average Louvain Modularity, Similarity, and Velocity Statistics per Month Pair')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_2018_2022_true_merged.png')
    plt.close()