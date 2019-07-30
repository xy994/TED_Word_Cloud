#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('pylab', '')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


df = pd.read_csv('df_with_clean_transcript')


# In[4]:


tags = ['3d printing', 'activism',       'Addiction', 'adventure', 'advertising', 'Africa', 'aging',       'agriculture', 'AI', 'AIDS', 'aircraft', 'algorithm',       'alternative energy', 'Alzheimers', 'ancient world', 'animals',       'animation', 'Anthropocene', 'anthropology', 'ants', 'apes',       'archaeology', 'architecture', 'art', 'Asia', 'asteroid',       'astrobiology', 'astronomy', 'atheism', 'augmented reality',       'Autism spectrum disorder', 'bacteria', 'beauty', 'bees',       'behavioral economics', 'big bang', 'big problems', 'biodiversity',       'Bioethics', 'biology', 'biomechanics', 'biomimicry', 'biosphere',       'biotech', 'birds', 'Blindness', 'blockchain', 'body language', 'books',       'botany', 'brain', 'Brand', 'Brazil', 'Buddhism', 'bullying',       'business', 'cancer', 'capitalism', 'cars', 'cello',       'charter for compassion', 'chemistry', 'children', 'china', 'choice',       'Christianity', 'cities', 'climate change', 'cloud', 'code',       'cognitive science', 'collaboration', 'comedy', 'communication',       'community', 'compassion', 'complexity', 'composing', 'computers',       'conducting', 'consciousness', 'conservation', 'consumerism',       'corruption', 'cosmos', 'creativity', 'crime', 'Criminal Justice',       'CRISPR', 'crowdsourcing', 'culture', 'curiosity', 'cyborg', 'dance',       'dark matter', 'data', 'death', 'Debate', 'decision-making',       'deextinction', 'demo', 'democracy', 'depression', 'design',       'dinosaurs', 'disability', 'disaster relief', 'discovery', 'disease',       'DNA', 'driverless cars', 'drones', 'ebola', 'ecology', 'economics',       'education', 'Egypt', 'empathy', 'energy', 'engineering',       'entertainment', 'entrepreneur', 'environment', 'epidemiology',       'Europe', 'evil', 'evolution', 'evolutionary psychology', 'exoskeleton',       'exploration', 'extraterrestrial life', 'extreme sports', 'failure',       'faith', 'family', 'farming', 'fashion', 'fear', 'feminism', 'film',       'finance', 'fish', 'flight', 'food', 'Foreign Policy', 'forensics',       'friendship', 'funny', 'future', 'gaming', 'garden', 'gender',       'Gender equality', 'Gender spectrum', 'genetics', 'geology', 'glacier',       'global development', 'global issues', 'goal-setting', 'God', 'Google',       'government', 'grammar', 'green', 'guitar', 'Guns', 'hack', 'happiness',       'health', 'health care', 'hearing', 'heart health', 'history', 'HIV',       'Human body', 'human origins', 'humanity', 'humor', 'identity',       'illness', 'illusion', 'immigration', 'india', 'industrial design',       'inequality', 'infrastructure', 'innovation', 'insects', 'intelligence',       'interface design', 'Internet', 'interview', 'introvert', 'invention',       'investment', 'Iran', 'iraq', 'Islam', 'jazz', 'journalism', 'language',       'law', 'leadership', 'LGBT', 'library', 'life', 'literature',       'live music', 'love', 'MacArthur grant', 'machine learning', 'magic',       'manufacturing', 'map', 'marketing', 'Mars', 'materials', 'math',       'media', 'medical imaging', 'medical research', 'medicine',       'meditation', 'meme', 'memory', 'men', 'mental health', 'microbes',       'microbiology', 'microfinance', 'microsoft', 'Middle East', 'military',       'mind', 'mindfulness', 'mining', 'mission blue', 'mobility',       'molecular biology', 'money', 'monkeys', 'Moon', 'morality',       'motivation', 'movies', 'museums', 'music', 'nanoscale', 'narcotics',       'NASA', 'natural disaster', 'Natural resources', 'nature',       'neuroscience', 'New York', 'news', 'Nobel prize', 'nonviolence',       'novel', 'nuclear energy', 'nuclear weapons', 'obesity', 'oceans',       'oil', 'online video', 'open-source', 'origami', 'pain', 'painting',       'paleontology', 'pandemic', 'parenting', 'peace', 'performance',       'performance art', 'personal growth', 'personality', 'pharmaceuticals',       'philanthropy', 'philosophy', 'photography', 'physics', 'physiology',       'piano', 'Planets', 'plants', 'plastic', 'play', 'poetry', 'policy',       'politics', 'pollution', 'population', 'potential', 'poverty',       'prediction', 'pregnancy', 'presentation', 'primates', 'prison',       'privacy', 'product design', 'productivity', 'programming',       'prosthetics', 'protests', 'psychology', 'PTSD', 'public health',       'public spaces', 'race', 'refugees', 'relationships', 'religion',       'resources', 'rivers', 'robots', 'rocket science', 'sanitation',       'science', 'science and art', 'security', 'self', 'Senses', 'sex',       'sexual violence', 'shopping', 'sight', 'simplicity', 'singer',       'skateboarding', 'Slavery', 'sleep', 'smell', 'social change',       'social media', 'society', 'sociology', 'software', 'solar energy',       'solar system', 'sound', 'South America', 'space', 'speech',       'spoken word', 'sports', 'state-building', 'statistics', 'storytelling',       'street art', 'String theory', 'student', 'submarine', 'success',       'suicide', 'Surgery', 'Surveillance', 'sustainability',       'synthetic biology', 'Syria', 'teaching', 'technology', 'TED Books',       'TED Brain Trust', 'TED en Español', 'TED Fellows', 'TED Prize',       'TED Residency', 'TED-Ed', 'TEDMED', 'TEDNYC', 'TEDx', 'TEDYouth',       'telecom', 'telescopes', 'television', 'terrorism', 'testing',       'theater', 'time', 'toy', 'trafficking', 'Transgender',       'transportation', 'travel', 'trees', 'trust', 'typography',       'United States', 'universe', 'urban', 'urban planning', 'Vaccines',       'violence', 'violin', 'virtual reality', 'virus', 'visualizations',       'vocals', 'vulnerability', 'war', 'water', 'weather', 'web',       'wikipedia', 'wind energy', 'women', 'women in business', 'work',       'work-life balance', 'world cultures', 'writing', 'wunderkind', 'youth']


# In[5]:


df_tags=df[tags]
tags_info = pd.DataFrame(df_tags.sum())


# In[6]:


for i in tags:
    tags_info.loc[i,'num_comment']=df[df[i]==1]['comments'].sum()
    tags_info.loc[i,'avg_duration']=round(df[df[i]==1]['duration'].mean()/60)
    tags_info.loc[i,'num_views']=df[df[i]==1]['views'].sum()
    tags_info.loc[i,'avg_views']=round(df[df[i]==1]['views'].mean())


# In[7]:


tags_info.head()


# In[8]:


tags_info.sort_values(by='num_views',ascending=False)


# In[9]:


top5 = tags_info.sort_values(by='num_views',ascending=False).head(5)
last = tags_info.sort_values(by='num_views',ascending=False).tail(5)


# In[35]:


#docs = list(df[df['art']==1]['clean_transcripts'])
docs = list(df['clean_transcripts'])
#instantiate CountVectorizer()
cv=CountVectorizer()
# this steps generates word counts for the words in your docs
word_count_vector=cv.fit_transform(docs)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
# count matrix
count_vector=cv.transform(docs)
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)


# In[36]:


feature_names = cv.get_feature_names()
important_words = []
for i in range(len(docs)):
    #get tfidf vector for first document
    first_document_vector=tf_idf_vector[i]
    #print the scores
    df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    #df.sort_values(by=["tfidf"],ascending=False).index
    #print(list(df.tfidf.nlargest(3).index))
    important_words.extend(list(df_t.tfidf.nlargest(3).index))
#print(important_words)


# In[37]:


from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(important_words))
figure(figsize=(20,10))
pyplot.imshow(wordcloud, interpolation='bilinear')
pyplot.axis("off")


# In[39]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#docs = list(df.clean_transcripts)
def tf_idf_words(tag,n):
    docs = list(df[df[tag]==1]['clean_transcripts'])
    #instantiate CountVectorizer()
    cv=CountVectorizer()
    # this steps generates word counts for the words in your docs
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # count matrix
    count_vector=cv.transform(docs)
    # tf-idf scores
    tf_idf_vector=tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()
    important_words = []
    for i in range(len(docs)):
        #get tfidf vector for first document
        first_document_vector=tf_idf_vector[i]
        #print the scores
        df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        #df.sort_values(by=["tfidf"],ascending=False).index
        #print(list(df.tfidf.nlargest(3).index))
        important_words.extend(list(df_t.tfidf.nlargest(n).index))
    return important_words


# In[42]:


for i in top5.index:
    words = tf_idf_words(i,5)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(words))
    figure(figsize=(20,10))
    title(i,fontdict={'fontsize':40})
    pyplot.imshow(wordcloud, interpolation='bilinear')
    pyplot.axis("off")


# ### Trend

# In[52]:


df2 = df[['event','clean_transcripts']]


# In[57]:


df2['year']=[x[-4:] for x in df2.event]


# In[59]:


df2.shape


# In[65]:


df2[df2.year.str.isdigit()].year.value_counts()


# In[62]:


df3 = df2[df2.year.str.isdigit()]


# In[63]:


df3.shape


# In[70]:


df4 = df3[df3.year.astype(int)>2009]


# In[71]:


df4.shape


# In[74]:


df5 = pd.get_dummies(df4,columns=['year'])


# In[75]:


df5.columns


# In[76]:


years = ['year_2010', 'year_2011', 'year_2012','year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017']


# In[78]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#docs = list(df.clean_transcripts)
def tf_idf_words(tag,n):
    docs = list(df5[df5[tag]==1]['clean_transcripts'])
    #instantiate CountVectorizer()
    cv=CountVectorizer()
    # this steps generates word counts for the words in your docs
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # count matrix
    count_vector=cv.transform(docs)
    # tf-idf scores
    tf_idf_vector=tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()
    important_words = []
    for i in range(len(docs)):
        #get tfidf vector for first document
        first_document_vector=tf_idf_vector[i]
        #print the scores
        df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        #df.sort_values(by=["tfidf"],ascending=False).index
        #print(list(df.tfidf.nlargest(3).index))
        important_words.extend(list(df_t.tfidf.nlargest(n).index))
    return important_words


# In[79]:


for i in years:
    words = tf_idf_words(i,3)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(words))
    figure(figsize=(20,10))
    title(i,fontdict={'fontsize':40})
    pyplot.imshow(wordcloud, interpolation='bilinear')
    pyplot.axis("off")


# In[120]:


docs = list(df5[df5['year_2017']==1]['clean_transcripts'])
#instantiate CountVectorizer()
cv=CountVectorizer()
# this steps generates word counts for the words in your docs
word_count_vector=cv.fit_transform(docs)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
# count matrix
count_vector=cv.transform(docs)
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()

first_document_vector=tf_idf_vector[1]
df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
#df.sort_values(by=["tfidf"],ascending=False).index
#print(list(df.tfidf.nlargest(3).index))


# In[105]:


for i in range(len(docs)):
    #get tfidf vector for first document
    first_document_vector=tf_idf_vector[i]
    #print the scores
    df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    #df.sort_values(by=["tfidf"],ascending=False).index
    #print(list(df.tfidf.nlargest(3).index))
    if i == 0:
        words_tfidf = pd.DataFrame(df_t.tfidf.nlargest(10)).reset_index()
        #print(words_tfidf)
    else:
        x=pd.DataFrame(df_t.tfidf.nlargest(10)).reset_index()
        words_tfidf=pd.concat([words_tfidf, x], axis=0)


# In[112]:


s = words_tfidf.groupby(words_tfidf['index']).mean()


# In[124]:


'well' in list(s.index)


# In[123]:


s.loc['well','tfidf']


# In[118]:


key_words = list(s.sort_values(by='tfidf',ascending=False).tfidf.nlargest(20).index)


# In[119]:


key_words


# In[125]:


def tf_idf_score(y,n):
    docs = list(df5[df5[y]==1]['clean_transcripts'])
    #instantiate CountVectorizer()
    cv=CountVectorizer()
    # this steps generates word counts for the words in your docs
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # count matrix
    count_vector=cv.transform(docs)
    # tf-idf scores
    tf_idf_vector=tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()

    first_document_vector=tf_idf_vector[1]
    df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    #df.sort_values(by=["tfidf"],ascending=False).index
    #print(list(df.tfidf.nlargest(3).index))
    for i in range(len(docs)):
        #get tfidf vector for first document
        first_document_vector=tf_idf_vector[i]
        #print the scores
        df_t = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        #df.sort_values(by=["tfidf"],ascending=False).index
        #print(list(df.tfidf.nlargest(3).index))
        if i == 0:
            words_tfidf = pd.DataFrame(df_t.tfidf.nlargest(n)).reset_index()
            #print(words_tfidf)
        else:
            x=pd.DataFrame(df_t.tfidf.nlargest(n)).reset_index()
            words_tfidf=pd.concat([words_tfidf, x], axis=0)
        s = words_tfidf.groupby(words_tfidf['index']).mean()
    col=[]
    for k in key_words:
        if k in list(s.index):
            col.append(s.loc[k,'tfidf'])
        else:
            col.append(0)
    return col


# In[132]:


dt_trend = pd.DataFrame(s.sort_values(by='tfidf',ascending=False).tfidf.nlargest(20))


# In[133]:


dt_trend.column=['year2017']


# In[134]:


dt_trend


# In[135]:


n = 10
for i in years:
    dt_trend[i]=tf_idf_score(i,n)


# In[140]:


dt_trend = dt_trend.drop(columns='tfidf')


# In[146]:


dt_trend.T.plot(figsize=(20,10))


# In[156]:


def word_count(y):
    docs = list(df5[df5[y]==1]['clean_transcripts'])
    s = ' '.join(docs)
    counts = dict()
    words = s.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


# In[157]:


x = word_count('year_2017')


# In[163]:


x.keys


# In[164]:


'born' in x


# In[159]:


x['born']


# In[168]:


df_wc = dt_trend.copy()


# In[169]:


for i in years:
    w = word_count(i)
    col=[]
    for k in key_words:
        if k in w:
            col.append(w[k])
        else:
            col.append(0)
    df_wc[i]=col


# In[171]:


df_wc.T.plot(figsize=(20,10))


# In[ ]:




