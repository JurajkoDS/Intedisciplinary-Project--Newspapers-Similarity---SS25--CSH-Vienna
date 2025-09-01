### Supervisors: Dr. Vito D. P. Servedio & Dr. Pietro Gravino
### Affiliation: TU Wien & Complexity Science Hub Vienna
### Author: Juraj Simkovic

## Description:
This project analyzes similarity between newspapers based on retweet patterns using Twitter data from 2018 to 2022. It explores how newspapers' audience behaviors change over time and what that reveals about the structure and dynamics of media ecosystems, particularly in Italy.

## Approach:
- Built a bipartite network of users and newspapers based on retweet activity.

- Projected the bipartite graph onto the newspaper layer to observe inter-newspaper relationships.

- Applied PCA as adimensionality reduction technique to reduce complexity and visualize node dynamics.

- Calculated and plotted Louvain modularity.

### Steps to generate the dataframe:
Using the load_data.ipynb notebook, the user can clean and merge the 5 (2018-2022) years of data that was available for analysis.
- It assumes the user has the data tweets_light.parquet, retweets_light.parquet and users_tw+rt_light.parquet, which should all be located in a 'data' folder, before the notebook can be run. The final dataframe is saved or loaded.
- The statistics dataframe is also generated using this notebook.

### Steps to generate the statistics table:
0. In step 0, we create a loop so that we can iterate over each of the 5 years.
1. We get a subset of the dataframe based on just 1 year.
2. The year is dissected into months.
3. For each month, we get a matrix that sums the retweets of individual leaders by individual followers.
4. Based on the matrix generated in step 3, calculate cosine similarities between leaders.
5. After we have the similarities, we map the leader names back to the columns.
6. All of the similarities are merged into a single dataframe called 'merged_similarities'. This dataframe contains NaN values, because not all leaders tweeted in all of the months. So if a leader has not tweeted in a specific month, it's similarity is missing.
7. Based on this 'merged_similarities' dataframe, statistical measures such as mean and standard deviation are calculated.
8. Afterwards, the velocities between individual consecutive months (i.e. January-February, February-March, etc.) are calculated. Based on these velocities, further statistical measures are calculated as well.
9. The Louvain algorithm for community detection and calculating average modularity based on consecutive months (just like in step 8) is added too.
10. Gather all the results into a single dataframe called stats_df.

### Further notebooks and .py files:
- the **modularity_timeseries+correlation.ipynb** generates a large timeseries modualarity graph, spanning over all 5 years (so all available data). It also generate a correlation heatmap, to see which statistical measures are significantly correlated.
- **script.py** generates all the graphs and .csv files that can be found in the individual year folders 2018-2022. It assumes 'df_checkpoint.parquet', which is the resulting dataframe generated in load_data.ipynb, can be found in the directory.
- **functions.py** is a collection of functions used in this project. It does contain all the functions used in this project, but most of them. Some of the other functions are in their respective notebooks or scripts.
