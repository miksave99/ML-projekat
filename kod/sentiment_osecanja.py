import pandas as pd #uvoz biblioteke za rad sa podacima

# Učitavanje podataka iz CSV fajla
data = pd.read_csv(r'tweet_emotions1.csv', encoding='latin-1',sep=',')
data.columns = ["tweet_id", "sentiment", "content"]
print(data.head())

df=pd.columns = ["tweet_id", "sentiment", "content"]


# drop unnecessary columns and rename cols
data.drop(['tweet_id'], axis=1, inplace=True)

data.columns = ['sentiment', 'content']

print(data.head())
data.to_csv(r'tweet_emotions1.csv')
# prebacivanje sentimenta u osecanja
sentiment_dict = {'boredom': 'negative',
                  'hate': 'negative',
                  'sadness': 'negative',
                  'anger': 'negative',
                  'worry': 'negative',
                  'relief': 'positive',
                'empty': 'neutral',
                  'happiness': 'positive',
                  'love': 'positive',
                  'enthusiasm': 'positive',
                  'neutral': 'neutral',
                  'surprise':'positive',
                  'fun': 'positive'
                 }
# nova kolona "polarity"
data['polarity'] = data.sentiment.map(sentiment_dict)
print(data.head())
data.to_csv(r'tweet_emotions2.csv')
