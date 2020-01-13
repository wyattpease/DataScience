from bs4 import BeautifulSoup
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

html_tables = {}

# Loading Html files into BeautifulSoup collection for analysis
for table_name in os.listdir('datasets'):
    table_path = f'datasets/{table_name}'
    table_file = BeautifulSoup(open(table_path, 'r'))
    html =  BeautifulSoup(open(table_path, 'r'))
    html_table = html.find(id='news-table')
    html_tables[table_name] = html_table

tsla = html_tables['tsla_22sep.html']
tsla_tr = tsla.findAll('tr')

# Grabbing the articles' text from their html elements
for i, table_row in enumerate(tsla_tr):
    # Read the text of anchor elements into 'link_text'
    link_text = table_row.a.get_text()
    # Read the text of the <td> elements into 'data_text'
    data_text = table_row.td.get_text()
    print(f'File number {i+1}:')
    print(link_text)
    print(data_text)
    if i == 3:
        break

parsed_news = []
# Putting the company, dates, times, and headlines of the articles into a list
for file_name, news_table in html_tables.items():
    for x in news_table.findAll('tr'):
        text = x.get_text() 
        date_scrape = x.td.text.split()
        # If the length of 'date_scrape' is 1, load 'time' as the only element
        # otherwise load date and time
        if len(date_scrape) == 1:
            time = date_scrape[0]
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Extracting the company ticker from the file name 
        ticker = file_name.split('_')[0]
        print(x.a.get_text())
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, x.a.get_text()])

new_words = {
    'crushes': 10,
    'beats': 5,
    'misses': -5,
    'trouble': -10,
    'falls': -100,
}

vader = SentimentIntensityAnalyzer()
vader.lexicon.update(new_words)

# Converting the parsed article text values into a DataFrame
columns = ['ticker', 'date', 'time', 'headline']
scored_news = pd.DataFrame(parsed_news, columns=columns)
scores = []

# Getting sentiment values for each article's headline and puting them in a dataframe
for item in scored_news['headline']:
    print(vader.polarity_scores(item))
    scores.append(vader.polarity_scores(item))
scores_df = pd.DataFrame(scores)

# Joining both dataframes
scored_news = pd.concat([scored_news,scores_df], axis=1).reindex(scored_news.index)
scored_news['date'] = pd.to_datetime(scored_news.date).dt.date
print(scored_news)

# Aggregating sentiment score means by date and company
mean_c = scored_news.groupby(['date', 'ticker']).mean()
mean_c = mean_c.unstack('ticker')
# Isolating the sentiment compound score 
mean_c = mean_c.xs("compound", axis=1)
mean_c.plot.bar()

# Counting the number of headlines before and after dropping duplicates
num_news_before = scored_news['headline'].count()
scored_news_clean = scored_news.drop_duplicates(subset=['ticker','headline'])
num_news_after = scored_news_clean['headline'].count()
f"Before we had {num_news_before} headlines, now we have {num_news_after}"

# Get data for a single day in order to plot the all of sentiment 
# scores for articles released on that day
single_day = scored_news_clean.set_index(['ticker', 'date'])
single_day = single_day.xs('fb', axis=0)
single_day = single_day['2019-01-03']
single_day['time'] = pd.to_datetime(single_day['time']).dt.time
single_day = single_day.set_index('time')
single_day = single_day.sort_index()
TITLE = "Positive, negative and neutral sentiment for FB on 2019-01-03"
COLORS = ["red","green", "orange"]

# Drop the columns that aren't useful for the plot and rename columns before plotting
plot_day = single_day.drop(['compound','headline'], axis=1)
plot_day.columns = ['negative','positive','neutral']
plot_day.plot.bar(stacked=True, figsize=(10,6))
plt.show()