# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('emails1.csv')
df=dataset.iloc[:,3:] # removing first three columns
df.columns
df.isnull().sum() #no na values
df.shape
df['Class'].value_counts()
df.info()
df.describe()
df.dtypes

# removing duplicate data
df.drop_duplicates(subset='content', keep='first', inplace=True)
df['Class'].value_counts()
df.describe()

fig1, ax1 = plt.subplots(figsize = (15,10))
ax1.pie(df['Class'].value_counts(), labels =['Non Abusive', 'Abusive'], autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


## How long are the lenght of the contents

df['length'] = df['content'].map(lambda text: len(text))
df['length'].describe()
sum(df['length']) # total words in emails


#This dataset is extremely skewed. The mean value is 1665.66 and yet the max length is 272036. Let's plot this on a logarithmic x-axis.
plt.xscale('log')
bins = 1.15**(np.arange(0,100))
plt.hist(df[df['Class']=='Non Abusive']['length'],bins=bins,alpha=0.8)
plt.hist(df[df['Class']=='Abusive']['length'],bins=bins,alpha=0.8)
plt.legend(('Non Abusive','Abusive'))
plt.show()

# Cleaning the texts
import re
import nltk

#data cleaning
def preprocessor(text):
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text

df['content']= df["content"].apply(preprocessor)

#tokenizing and lemmatizing
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer() 

def tokenizer_lemmatizer(text):
    return[lem.lemmatize(word, "v") for word in text.split()]


from nltk.corpus import stopwords
stop= stopwords.words('english')
stop=stop + ['john', 'j', 'lavorato','subject', 'excelr', 'pm', 'john', 'arnold', 'hou', 'ect', 'cc', 'bc', 'eat','pm','arnold','hou','ect','cc','subject','football','bet','minn','phil',
 'indi','cinnci','det','clev','den','dall','jack','gentleman','approximate','retail','price',
 'interest','trading','red','derived','spec','website','winesearcer','ha','stored','temperature',
 'controlled','wine','storage','facility','quan','vintage','perrier','jouet','brut',
 'fleur','de','champagne','piper','heidsek','reserve','http','www','asp','final','subject','e','hour','cd',
 'folder','synchronizing','day','time','quantity','back','u','found','td', 'br', 'tr', 'sc', 
'fool','cut','woman','company','year','detail','trans_type','mkt_type','delivery','data','original','engy','free','good','texas','man','space','type','call']

#importing new more stopwords
stop_words = []
with open("stop.txt") as f:
    stop_words = f.read()

# Convert stopwords to list
def Convert(string): 
    li = list(string.split("\n")) 
    return li
s_2=Convert(stop_words)

#updating list of stopwords and saving into sr_1
stop=stop+s_2


#creating total corpus of mails
corpus = []

for i in df.index.values:
    mail_content=[w for w in tokenizer_lemmatizer(df['content'][i]) if w not in stop]
    
    # lem = WordNetLemmatizer()
    # df['content'] = [lem.lemmatize(word, "v") for word in df['content'] if not word in set(stop)]
    mail_content = ' '.join(mail_content)
    corpus.append(mail_content)
    

# creating new cleaned dataset
new_df=pd.DataFrame(list(zip(corpus, list(df['Class']))), columns=['content', 'Class']) 

pd.DataFrame(new_df).to_csv("email_cleaned.csv",encoding="utf-8")

# Joinining all the reviews into single paragraph 
corpus_string = " ".join(corpus)
#all word tokens
words_tokens= word_tokenize(corpus_string)

corpus_words = corpus_string.split(" ")

# Only non- abusive and abusive mails
non_abusive = new_df.content[new_df.Class=='Non Abusive'].sample(frac=1, random_state=42)
non_abusive.head()
non_abusive.size

abusive = new_df.content[new_df.Class=='Abusive'].sample(frac=1,random_state=42)
abusive.head()
abusive.size

#abusive and non-abusive word tokens
abusive_string = " ".join(abusive)
abusive_tokens= word_tokenize(abusive_string)

nonabusive_string = " ".join(non_abusive)
non_abusive_tokens= word_tokenize(nonabusive_string)

# ### Counter

from collections import Counter

counter = Counter(words_tokens)
counter.most_common(20)


non_abusive_counter = Counter(non_abusive_tokens)
non_abusive_counter

abusive_counter = Counter(abusive_tokens)
abusive_counter

# # Bar Plot
# =============================================================================
# convert list of tuples into data frame
freq_df = pd.DataFrame.from_records(counter.most_common(20), columns =['Words_Token','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='Words_Token', figsize=(15,10),fontsize=15);

# Bar plot for Non Abusive mails
non_ab_freq_df = pd.DataFrame.from_records(non_abusive_counter.most_common(20), columns =['Non_abusive Token','Count'])
#Creating a bar plot
non_ab_freq_df.plot(kind='bar',x='Non_abusive Token', figsize=(15,10), fontsize=15)
plt.xticks(rotation=45);

# Bar plot for Abusive mails
ab_freq_df = pd.DataFrame.from_records(abusive_counter.most_common(40), columns =['abusive Token','Count'])
#Creating a bar plot
ab_freq_df.plot(kind='bar',x='abusive Token', figsize=(15,10), fontsize=15)
plt.xticks(rotation=45);

# Simple word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_tot = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_string)

plt.imshow(wordcloud_tot)

# wordcloud for abusive 
wordcloud_abusive = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(abusive_string)

plt.imshow(wordcloud_abusive)

# wordcloud for non-abusive
wordcloud_nonabusive = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(nonabusive_string)

plt.imshow(wordcloud_nonabusive)


# positive words # Choose the path for +ve words stored in system
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# negative words  Choose path for -ve words stored in system
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
corpus_neg_in_neg = " ".join ([w for w in corpus_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
corpus_pos_in_pos = " ".join ([w for w in corpus_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)


# Unique words 
unique_words = list(set(" ".join(corpus_words).split(" ")))