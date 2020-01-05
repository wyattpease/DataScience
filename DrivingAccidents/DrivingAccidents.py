from sklearn import linear_model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#This project it to search for a pattern in traffic accidents across states through 
#data wrangling, plotting, dimensionality reduction, and unsupervised clustering.

#Read in and examine the data
car_acc = pd.read_csv('datasets/road-accidents.csv', comment='#', sep ='|')

rows_and_cols = car_acc.shape
print('There are {} rows and {} columns.\n'.format(
    rows_and_cols[0], rows_and_cols[1]))

car_acc_information = car_acc.info()
print(car_acc_information)

car_acc.tail()


# Compute the summary statistics of all columns in the dataframe
sum_stat_car = car_acc.describe()
print(sum_stat_car)

# Create a pairwise scatter plot to explore the data
sns.pairplot(car_acc)

# Can see a relationship between the number of fatal accidents and the remaining three value types
# Correlation coefficient to quantify these relationships
corr_columns = car_acc.corr()
corr_columns

# Can see some of the reaming three columns have a positive correlation with each other, 
# using multivariate linear regression to isolate each variable's correlation to the 'drvr_fatl_col_bmiles' outcome
features = car_acc[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
target = car_acc['drvr_fatl_col_bmiles']

reg = linear_model.LinearRegression()

# Fit a multivariate linear regression model
reg.fit(features, target)

fit_coef = reg.coef_
print('Regression Coefficients: ',fit_coef)

# Standardizing feature columns in order to perform PCA and show proportion of variance explained
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA()
pca.fit(features_scaled)

# Plot the proportion of variance explained on the y-axis of the bar plot
plt.figure(1)
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])
plt.title('Proportion of Variance Explained')

# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_.cumsum()[1]
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 5)))

# Transform the standardized features using two principal components
pca = PCA(n_components=2)
p_comps = pca.fit_transform(features_scaled)

# Extract the first and second component to use for the scatter plot
p_comp1 = p_comps[:,0]
p_comp2 = p_comps[:,1]

plt.figure(2)
# Scatter to visualize the first 2 principal components
plt.scatter(p_comp1,p_comp2)
plt.title('Top 2 Principal Components in road-accidents dataset')
#Not very clear, going to use KMeans to see when clustering stops adding explanatory power
plt.figure(3)
plt.title('KMeans Clusters')
ks = range(1, 10)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=8)
    km.fit(features_scaled)
    inertias.append(km.inertia_)

plt.plot(ks, inertias, marker='o')

km = KMeans(n_clusters=3, random_state=8)
km.fit(features_scaled)

# Create a scatter plot of the first two principal components to examine any potential patterns
plt.figure(4)
plt.scatter(p_comps[:,0], p_comps[:,1], km.labels_)
plt.title('Top 2 Principal Components Scatter')

#return to unstandardized features to highlight any differences between the clusters in the feature columns
car_acc['cluster'] = km.labels_

# Reshape the DataFrame to the long format
melt_car = pd.melt(car_acc, id_vars= 'cluster', var_name='measurement', value_name='percent', value_vars=features)

# Create a violin plot splitting and coloring the results according to the km-clusters
plt.figure(5)
plt.title('Results by Cluster Violin')
sns.violinplot(x=melt_car['percent'], y=melt_car['measurement'],hue=car_acc['cluster'])
miles_driven = pd.read_csv('datasets/miles-driven.csv', sep='|')

car_acc_miles = car_acc.merge(miles_driven, on='state')
car_acc_miles['num_drvr_fatl_col'] = car_acc_miles['drvr_fatl_col_bmiles']/1000 * car_acc_miles['million_miles_annually']

# Create a barplot of the total number of accidents per cluster
plt.figure(6)
plt.title('Total Accidents per Cluster')
sns.barplot(x=km.labels_, y=car_acc_miles['num_drvr_fatl_col'], data=car_acc, estimator=sum, ci=None)
plt.show()
# Calculate the number of states in each cluster and their 'num_drvr_fatl_col' mean and sum.
count_mean_sum = car_acc_miles.groupby('cluster').agg({'state':'count','num_drvr_fatl_col':['mean', 'sum']})
print(count_mean_sum)