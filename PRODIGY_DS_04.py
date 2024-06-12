#!/usr/bin/env python
# coding: utf-8

# ## TASK 4 
# 
# ### AIM :
# 
#  To analyze sentiment patterns in Twitter data, visualize the distribution of sentiments, and identify the most frequent words and entities in the tweet comments.

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px


# In[3]:


# Load the dataset
file_path = 'twitter_training.csv'
df = pd.read_csv(file_path)


# In[4]:


# Display the first few rows of the dataset
print(df.head())


# In[5]:


# Display the original column names
print("Original Column Names:")
print(df.columns)


# In[6]:


# Rename the columns
new_column_names = {'2401': 'Twitter_id', 'Borderlands': 'Entities', 'Positive':'Sentiments','im getting on borderlands and i will murder you all ,':'Tweetcontents'}
df = df.rename(columns=new_column_names)


# In[7]:


# Display the new column names
print("\nNew Column Names:")
print(df.columns)


# In[8]:


#dropping null values
df.dropna(inplace=True)
df


# In[9]:


print(df.head())
#sentiment distribution
sentiment_counts=df['Sentiments'].value_counts()


# In[10]:


#Sentiment distribution
plt.figure(figsize=(8,6))
sns.countplot(x="Sentiments",data=df,order=df["Sentiments"].value_counts().index)
plt.title("Sentiment Distribution")
plt.xlabel('Sentiments')
plt.ylabel("Number of Reviews")
plt.show()


# ##### The sentiment distribution shows that there are more positive tweets compared to negative and neutral ones.
# 

# In[25]:


#Word cloud for positive sentiments

from wordcloud import WordCloud


# In[26]:


# Filter the dataset to include only tweets with positive sentiment
positive_tweets = df[df['Sentiments'] == 'Positive']

# Concatenate all positive tweets into a single string
positive_text = ' '.join(positive_tweets['Tweetcontents'])

# Create a WordCloud object
wordcloud = WordCloud(width=200, height=100, background_color='lightblue').generate(positive_text)


# In[27]:


# Display the WordCloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Positive Sentiments')
plt.axis('off')
plt.show()


# In[28]:


#Word cloud for positive sentiments
from wordcloud import WordCloud


# In[29]:


# Filter the dataset to include only tweets with positive sentiment
positive_tweets = df[df['Sentiments'] == 'Negative']

# Concatenate all positive tweets into a single string
positive_text = ' '.join(positive_tweets['Tweetcontents'])

# Create a WordCloud object
wordcloud = WordCloud(width=200, height=100, background_color='pink').generate(positive_text)


# In[31]:


# Display the WordCloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Negative Sentiments')
plt.axis('off')
plt.show()


# In[32]:


#Word cloud for neutral sentiments

from wordcloud import WordCloud


# In[33]:


# Filter the dataset to include only tweets with positive sentiment
positive_tweets = df[df['Sentiments'] == 'Neutral']

# Concatenate all positive tweets into a single string
positive_text = ' '.join(positive_tweets['Tweetcontents'])

# Create a WordCloud object
wordcloud = WordCloud(width=200, height=100, background_color='lightgreen').generate(positive_text)


# In[34]:


# Display the WordCloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Neutral Sentiments')
plt.axis('off')
plt.show()


# ##### The word clouds provide insights into the most frequent words associated with each sentiment category.
# 

# In[35]:


Sentiment_counts=df['Sentiments'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(Sentiment_counts,labels=Sentiment_counts.index,autopct='%1.1f%%',startangle=140)
plt.title('Distribution of Sentiments')
plt.show()


# ##### The pie chart illustrates the distribution of sentiments, indicating the proportion of each sentiment category in the dataset.
# 

# In[36]:


#bar plot of top 10 frequent words in tweet comments
from sklearn.feature_extraction.text import CountVectorizer
vectorizer= CountVectorizer(stop_words='english',max_features=10)
word_frequency=vectorizer.fit_transform(df['Tweetcontents'])
words=vectorizer.get_feature_names_out()
word_counts=word_frequency.sum(axis=0).A1


# In[37]:


plt.figure(figsize=(8,6))
sns.barplot(x=word_counts, y=words,palette='dark')
plt.title('Top 10 frequent words in Tweet comments')
plt.xlabel("Word_count")
plt.ylabel("Words")
plt.show()


# ##### The bar plot of the top 10 frequent words gives a clear view of the most common words used in tweet comments.
# 

# In[38]:


fig=px.pie(df,names='Entities',hole=0.4,title="Doughnut for Entities")
fig.show()


# * The doughnut chart provides an overview of the distribution of entities in the dataset.
# * Overall, this analysis helps understand the sentiment patterns and common topics discussed in the Twitter dataset.
