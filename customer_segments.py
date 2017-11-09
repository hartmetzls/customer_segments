
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
# get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:

# display(data.head(n=3))

# Display a description of the dataset
display(data.describe())

data['Total_spending_by_vendor'] = data.sum(axis=1)
display(data.head())
display(data.describe())
data = data.drop(['Total_spending_by_vendor'], axis=1)
display(data.head())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [378, 436, 62]

# # Generates a set of three customers. Customers that varied were entered into the indices list.
# from random import randint
# max = len(data)
# for i in range(0, 3):
#     indices.append(randint(0, max))
# print indices

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)
# print type(samples)

#Shows row sum for each sample for question below
samples['Total_Spending'] = samples.sum(axis=1)
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 3071.9
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

# ----------------------------------------------------------------------------------------------------------
# 
# The above mean values were provided in the notebook, but Frozen is missing and it looks like the Grocery number is incorrect. (Please correct me if I'm wrong.) The means I observe are as follows:
#     
# * Fresh: 12000.3
# * Milk: 5796.2
# * Grocery: 7951.3
# * Frozen: 3071.9
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8

# **Answer:**
# 
# For this question, since we don't know exactly what each of these categories include, I'll continue with the assumption that Fresh is produce, Grocery is foods with a long shelf life, home products, and toiletries, and Delicatessen is meats with a shorter shelf life.
# 
# -  #### 0th establishment: 
# Below average for all categories and Total_spending. Looking at the Total_spending column I added to the dataframes, it's between the min and 25th percentile. These observations make me guess that it's a small business, maybe a small coffee shop, seeing as how their biggest spending is on Milk and Fresh.
# -  #### 1st establishment: 
# Above average for Fresh, Frozen, Delicatessen, and Total_spending. Below for Milk, Grocery, and Detergents_paper. It's above the 75th percentile for Total_spending, so based on that and its category spending, I'm guessing this is a large restaurant.
# -  #### 2nd establishment: 
# Below average for Fresh, Grocery, Delicatessen, and Total_spending. Above average for Milk, Frozen, and  Detergents_paper. Looking at the Total_spending column, it places between the 50th and 75th percentiles. These observations lead me to think this is a moderately sized market/bodega.
# 
# These comparisons did help drive my insights into what kind of establishments they might be.

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[4]:

for feature in data:
    # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop([feature], axis=1)

    # TODO: Split the data into training and testing sets(0.25) using the given feature as the target
    # Set a random state.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data, 
                                                        data[feature], 
                                                        test_size = 0.25, 
                                                        random_state = 2)

    from sklearn.tree import DecisionTreeRegressor
    scores = []
    for i in range(1, 101):
        # TODO: Create a decision tree regressor and fit it to the training set
        regressor = DecisionTreeRegressor(random_state=i)
        regressor.fit(X_train, y_train)
        regressor.predict(X_test)

        # TODO: Report the score of the prediction using the testing set
        score = regressor.score(X_test, y_test)
        # print "R^2 score using random_state {}: {}".format(i, score)
        scores.append(score)
    avg_score = np.average(scores)
    print "{}: {}".format(feature, avg_score)
    
# Resource:
# https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-pandas-dataframe-to-run-regression


# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to believe that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.

# **Answer:**
# 
# -  For this answer, I'll comment on the attempt to predict Delicatessen purchasing.
# -  The reported prediction score was -2.09.
# -  The R^2 implies the model fails to fit the data, which leads me to believe the feature point is hard to predict using the other features, and thereby is necessary for identifying customers' spending habits.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[5]:

#https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
import matplotlib.pyplot as plt
import seaborn
data.hist(bins=30, figsize = (14, 8))
plt.show()

# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

seaborn.heatmap(data.corr())


# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)(the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

# **Answer:**
# 
# -  I printed some histograms to confirm some of my observations from the scatter matrix. It seems all features are unimodal, assymetric, and right-skewed. All have outliers (with significantly higher values) and have a large number of data points near 0. 
# -  Some pairs of features exhibit some degree of correlation. Detergent_Paper and Grocery together indicate moderate dependence. Milk and Grocery indicate weak dependence, as do Milk and Detergent_Paper.
# -  This confirms my suspicions about the relevance of the Delicatessen - that it is likely relevant.
# -  The Delicatessen data is extremely unimodal, assymetric, and right-skewed, with most values near zero and one or more extreme outliers.
# 
# Resource:
# 
# http://www.statisticshowto.com/probability-and-statistics/skewed-distribution/  
# 

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[6]:

# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
samples = samples.drop('Total_Spending', 1)
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Resource:
# https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe-using-python-del  


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[7]:

# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[8]:

# For each feature find the data points with extreme high or low values
all_outliers = []
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1)*1.5
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    # Create a dataframe of the potential outliers
    all_outliers_df = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    
    # Append indices to list
    all_outliers.append(all_outliers_df.index.values.tolist())

# Flatten list
all_outliers = [val for sublist in all_outliers for val in sublist]
    
multivariate_outliers  = []
for i in all_outliers:
    if all_outliers.count(i) > 1:
        multivariate_outliers.append(i)
        while all_outliers.count(i) > 0:
            all_outliers.remove(i)
print "Multivariate outliers:"
print multivariate_outliers

# OPTIONAL: Select the indices for data points you wish to remove
outliers = multivariate_outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# Resources:
# https://stackoverflow.com/questions/11264684/flatten-list-of-lists
# https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item-in-python
# https://stackoverflow.com/questions/43772362/how-to-print-a-specific-row-of-a-pandas-dataframe


# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.

# **Answer:**
# 
# -  Yes, the data points at the "data" dataframe indices 65, 66, 75, 128, and 154 had data points considered outliers for more than one feature.
# -  Yes, at least the multivariate outlier should be removed. My initial run of this project revealed that leaving the outliers for multiple categories resulted in an unclear optimal cluster size. I decided not to remove all of the outliers because based on the above definition of an outlier, about 5% of our data points are outliers. If that percentage were smaller, I would think the outliers were data entry errors, but since there are so many, there's a chance they're important to include in our dataset.
# 
# K-Means does suffer from the presence of outliers. Because of this, I ultimately removed some to help reveal an optimal cluster size.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[9]:

from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Resource:
# https://stackoverflow.com/questions/20297332/python-pandas-dataframe-retrieve-number-of-columns


# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:**
# 
# -  0.7068
# -  0.4430, 0.2638, 0.1231, and 0.1012 respectively. 0.9311 in total.
# -  Bearing in mind that these dimensions are sorted from most to least explained variance, this visualization shows that in [given] dimension, the axis (or eigenvector) created by combining all features with [these] weights represent the direction of maximum variance.
#     -  Dimension 1: This gives our highest sole explained variance. Here we give the most extreme weights to spending on Detergents_Paper, Grocery, and Milk. They are negatively correlated with Fresh and Frozen (i.e. They have an inverse relationship with Fresh and Frozen). This might represent categories that are frequently purchased together.
#     -  Dimension 2: The cumulative variance is 0.7068. All features get weighted negatively, but mostly spending on Fresh, Delicatessen, and Frozen. With respect to this dimension, all variables are correlated to some degree.
#     -  Dimension 3: The cumulative variance is 0.8299. Here we give the most extreme weights, again, to spending on Fresh, Delicatessen, and Frozen. Fresh is inversely correlated with Delicatessen and Frozen. This might represent that Delicatessen and Frozen are purchased together, but not with Fresh.
#     -  Dimension 4: The cumulative variance is 0.9311. Here we give the most extreme weights, again, to spending on Frozen, Delicatessen, and Fresh. In this dimension, we see that there's an inverse relationship between Frozen vs. Delicatessen and Fresh spending, showing that Frozen spending tends to correlate with lower Delicatessen and Fresh spending.
#     -  Dimension 5: The cumulative variance is 0.9796. The most extreme weights go to spending on Milk, Detergents_Paper, and Grocery.
#     -  Dimension 6: The culumative variance is 1. The most extreme weights are given to spending on Grocery, Milk, and Detergents_Paper.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[10]:

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[11]:

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[12]:

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[13]:

# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.

# **Answer:**
# -  #### K-Means clustering advantages:
#     - Easy to implement 
#     - Can be used for market segmentation
#     - Well suited for a small dataset with an unknown number of clusters
#     - Will always converge
# - #### Gaussian Mixture Model clustering advantages:
#     - Well suited for a small dataset with an unknown number of clusters
#     - At least one version is well suited for small datasets with an unknown number of categories
#     - Gives a probability output (continuous) rather than discrete (soft clustering)
#     - More flexible than K-means clustering, allowing for elliptical clusters
# - A Gaussian Mixture Model seems more appropriate for this dataset. My guess at this point is that this dataset does not have clear clusters, so we will benefit from starting with a method that has options to use soft clustering, like GMM or Bayesian GMM.
#     
# 
# Resources:
# 
# https://en.wikipedia.org/wiki/K-means_clustering   
# http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html   
# https://youtu.be/JNlEIEwe-Cg
# 

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[14]:

# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
n_components_range = range(2, 7)
# print n_components_range
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=6)
    clusterer = gmm.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)

    print "Silhouette score for {} components: {}   ".format(n_components, score)
    
gmm = GaussianMixture(n_components=2, random_state=6)
clusterer = gmm.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.means_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Resources:
# https://github.com/scikit-learn/scikit-learn/issues/7623
# https://stackoverflow.com/questions/15286401/print-multiple-arguments-in-python


# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?

# **Answer:**
# -  Silhouette score for 2 components: 0.421916846463   
# Silhouette score for 3 components: 0.366083165806   
# Silhouette score for 4 components: 0.318959619067   
# Silhouette score for 5 components: 0.297898612498   
# Silhouette score for 6 components: 0.296603538245   
# -  A cluster number of 2 has the best silhouette score.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[15]:

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[16]:

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers['Total_Spending'] = true_centers.sum(axis=1)
true_centers.index = segments
display(true_centers)

# Displays means
print "Mean values for dataset:"
mean = pd.DataFrame(data.mean(axis=0))
display(mean.T)


# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

# **Answer:**
# - A customer assigned to Cluster 0 should best identify with the establishments represented by Segment 0. Comparing Segment 0 customers to the mean values for the dataset, they're high in Fresh spending, and low in Milk, Grocery, and Detergents_Paper spending. Considering that the model identified only two clusters, any guess is likely an overgeneralization. My best guess is that this cluster represents restaurants (or other food service establishments) which are of small to moderate size.
# - A customer assigned to Cluster 1 should best identify with the establishments represented by Segment 1. Comparing Segment 1 customers to the mean values for the dataset, they're high in Milk, Grocery, and Detergents_Paper spending, and low in Fresh and Frozen spending. My best guess is that this cluster represents markets and cafes that are moderate to large in size.

# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[17]:

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred, "   "
    
display(true_centers)

print "Sample customers:"
samples['Total_Spending'] = samples.sum(axis=1)
display(samples)


# **Answer:**
# - Sample point 0 predicted to be in Cluster 0    
# Sample point 1 predicted to be in Cluster 0    
# Sample point 2 predicted to be in Cluster 1    
# - The predictions made by the clusterer are fairly consistent with my own predictions in Question 1. The 0th establishment's relative size guess was consistent and it seems it does indeed serve food. The 1st establishment's size guess was consistent, as was the guess that it's a restaurant. The 2nd establishment's size guess was consistent, as was the guess that it is a market. Referring back to the cluster visualization, we can see that this establishment is right on the edge of both clusters. (Interestingly enough, if the clusterer is run without removing the multivariate outliers, the clusterer will predict sample point 2 to be in Cluster 0.) 
# 
# Resource:
# https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:**
# - We cannot assume the change will affect all customers equally. The wholesale distributor could take n customers from Cluster 0 and n customers from Cluster 1 to test the 3-day delivery week. At the end of this A/B test, the distributor can survey the customers from each cluster who were switched to a 3-day delivery week and assess their satisfaction. My guess is that would be a negative change for Cluster 0 because they rely heavily on Fresh foods, and that it would be fine for most Cluster 1 customers.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:** 
# - We could use our existing clustered data points to generate a supervised learner. This supervised model would have the target variables (labels) Cluster 0 or Cluster 1, and the category spending as features. The model could take in the new customers' category spending and predict the cluster in which they fit. Alternatively, we could create a model that would take in customers' business type (Hotel/Restaurant/Cafe or Retailer) and predict their spending in each category. 

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[18]:

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?

# **Answer:**
# - The clustering algorithm is not far off, although it does classify a number of Retailers with the Hotel/Restaurant/Cafe cluster. The algorithm does choose the correct numbers of clusters.
# - Using the Channel feature, the distribution classifies exactly two segments: 'Retailers' and 'Hotels/Restaurants/Cafes'. Using the GMM clustering model, these segments are classified in a very generalized way. While the centroids are the same, the GMM clustering without the Channel feature does not identify Hotel/Restaurants/Cafes or Retailers far from their respective centroids.
# - Mostly yes. I guessed that Cluster 0 (on the right) was made up of food service establishments. I guessed that Cluster 1 (on the left) was markets/retailers and cafes. It's certainly true that they're mostly markets. It's possible that the red data points closer to the Cluster 0 centroid are indeed cafes. More details from the distributor would reveal whether or not that's the case. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ## Note
# 
# Portions of this code were provided by Udacity. To see which portions were provided, follow the link below.
# 
# https://github.com/udacity/machine-learning/blob/master/projects/customer_segments/customer_segments.ipynb
# 
# Resource note: The Udacity reviewers were resources for various parts of this project.

# In[ ]:



