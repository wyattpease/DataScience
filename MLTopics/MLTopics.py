import pandas as pd
import matplotlib.pyplot as plt
import re
import wordcloud as wc
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
from sklearn.decomposition import LatentDirichletAllocation as LDA

#Analyzing Zika outbreaks using NLP methods

papers = pd.read_csv('Datasets/paho.csv')
print(papers.head())

#drop columns that do not contain useful information
papers = papers.drop(['Month of Date','Report Epi Week', 'Year of Date'], axis=1)
print(papers.head())

#Determine number of reports per year
groups = papers.groupby('Date')
counts = groups.size()
plt.figure()
plt.plot(counts)

#pre-processing text to convert to Uppercase 
#print the first 5 rows before and after to confirm the changes
print(papers['Country / territory'].head())
papers['Country_processed'] = papers['Country / territory'].map(lambda x: re.sub('[,\.!?]', '', x))
papers['Country_processed'] = papers['Country_processed'].map(lambda x: x.upper())
print(papers['Country_processed'].head())

#wordcloud to visualize data
long_string = " ".join(papers['Country_processed'])
wordcloud = wc.WordCloud()
wordcloud.generate(long_string)
wordcloud.to_image()

#prepare for LDA
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    
    plt.figure()
    plt.bar(x_pos, counts, align='center')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.title('Countries most selected for study')
    plt.plot()
    plt.show()

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(papers['Country_processed'])
plot_10_most_common_words(count_data, count_vectorizer)

#LDA trend analysis
warnings.simplefilter("ignore", DeprecationWarning)

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

number_topics = 1
number_words = 15

lda = LDA(n_components=number_topics)
lda.fit(count_data)

print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
