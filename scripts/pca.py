import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

# Load data and apply PCA
df_copy = pd.read_csv('/Users/minghaotian/Desktop/features_new.csv')
X = df_copy.drop('label', axis=1)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)

# Create DataFrame for the principal components
principalDf = pd.DataFrame(data=principalComponents, 
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])

# Concatenate with labels
finalDf = pd.concat([principalDf, df_copy[['label']]], axis=1)

# Create a 3D plot with smaller data points
fig = px.scatter_3d(finalDf, x='principal component 1', y='principal component 2', 
                    z='principal component 3', color='label')

# Adjust marker size
fig.update_traces(marker=dict(size=3.5, line=dict(width=0)))

fig.show()
