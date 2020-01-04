# Import libraries
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
# Load the data
df = pd.read_csv('datasets/cosmetics.csv')

# Check the first five rows 
print(df.head())

# Inspect the types of products
df['Label'].value_counts()

# Filter for moisturizers
moisturizers = df.loc[df['Label'] == 'Moisturizer']

# Filter for dry skin as well
moisturizers_dry = moisturizers.loc[moisturizers['Dry'] == 1]

# Reset index
moisturizers_dry = moisturizers_dry.reset_index(drop = True)

# Initialize dictionary, list, and initial index
ingredient_idx = {}
corpus = []
idx = 0

# For loop for tokenization
for i in range(len(moisturizers_dry)):    
    ingredients = moisturizers_dry['Ingredients'][i]
    ingredients_lower = ingredients.lower()
    tokens = ingredients_lower.split(', ')
    corpus.append(tokens)
    for ingredient in tokens:
        if ingredient not in ingredient_idx:
            ingredient_idx[ingredient] = idx
            idx += 1
            
# Check the result 
print("The index for decyl oleate is", ingredient_idx['decyl oleate'])

# Get the number of items and tokens 
M = len(moisturizers_dry)
N = len(ingredient_idx)

# Initialize a matrix of zeros
A = np.zeros((M,N))

# Define the oh_encoder function
def oh_encoder(tokens):
    x = np.zeros((N))
    for ingredient in tokens:
        # Get the index for each ingredient
        idx = ingredient_idx[ingredient]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x

# Make a document-term matrix
i = 0
for tokens in corpus:
    A[i, :] = oh_encoder(tokens)
    i += 1

# Dimension reduction with t-SNE
model = TSNE(n_components = 2, learning_rate = 200, random_state = 42)
tsne_features = model.fit_transform(A)

# Make X, Y columns 
moisturizers_dry['X'] = tsne_features[:,0]
moisturizers_dry['Y'] = tsne_features[:,1]

from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

# Make a source and a scatter plot  
source = ColumnDataSource(moisturizers_dry)
plot = figure(x_axis_label = 'TSNE 1', 
              y_axis_label = 'T-SNE 2', 
              width = 500, height = 400)
plot.circle(x = 'X', 
    y = 'Y', 
    source = source, 
    size = 10, color = '#FF7373', alpha = .8)

# Create a HoverTool object
hover = HoverTool(tooltips = [('Item','@Name'),
                              ('Brand','@Brand'),
                              ('Price','$@Price'),
                              ('Rank','@Rank')])
plot.add_tools(hover)

show(plot)
