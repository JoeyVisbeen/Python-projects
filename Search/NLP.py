import nltk
from nltk.corpus import reuters
from nltk.stem.wordnet import WordNetLemmatizer 
#from nltk.stem.porter import PorterStemmer 


# text Pre-processing

noise_list = ["We","we","Dit","dit","Een","een","Om","om","Te","te","Een","een","De","de"]
# very basic misses alot, like to lowercase
def _remove_noise_str(input_text: str):
    words = input_text.split()
    noise_free_words  = [word for word in words if word not in noise_list]
    return " ".join(noise_free_words)

#def _remove_noise_arr(input_array: list):
#    lower_array = []
#    for i in input_array:
#        lower_array.append(i.lower())

#    noise_free_words  = [word for word in lower_array if word not in noise_list]
#    return " ".join(noise_free_words)

#print(_remove_noise_str("Dit is een text om te testen."))
#print(_remove_noise_arr(["Goedemorgen", "wat", "is", "dit", "een", "mooie", "dag!"]))

# Lexicon normalization Lem dont work?

#lem = WordNetLemmatizer()
#stem = PorterStemmer()
#word = "Multiplying"
#lem.lemmatize(word, "v")
#print(word)
#print(stem.stem(word))

# object standardization

lookup_dict = {"rt":"retweet", "dm":"direct message", "awsm":"awesome", "luv":"love", "#":"hashtag:", "@":"accountlink:"}

def _clean_punctuation(word_array):
    for i in range(len(word_array)):
        if word_array[i]

def _lookup_words(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        
        lowerWord = word.lower()
        lowerWordChars = list(lowerWord)

        if lowerWord in lookup_dict:
            word = lookup_dict[word.lower()]
        
        if lowerWordChars[0] in lookup_dict:
            prefix = lookup_dict[lowerWord[0]]
            lowerWordChars.pop(0)
            
            word = prefix + "".join(lowerWordChars)
        
        new_words.append(word) 
    
    new_text = " ".join(new_words)
    return new_text

#print(_lookup_words("RT this is a retweeted twee by Donald Trump"))

#from nltk import word_tokenize, pos_tag
#text = "book my flight, I will read this book"
#tokens = word_tokenize(text)
#print(pos_tag(tokens, tagset="universal"))


#doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
#doc2 = "My father spends a lot of time driving my sister around to dance practice."
#doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
#doc_complete = [doc1, doc2, doc3]
#doc_clean = [doc.split() for doc in doc_complete]
#print (doc_clean)
#Topic modeling
#import gensim 

#dictionary = gensim.corpora.Dictionary(doc_clean)
#print(dictionary)
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#print(doc_term_matrix)
#Lda = gensim.models.ldamodel.LdaModel

#ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
#print(ldamodel.print_topics())



from nltk.tokenize import word_tokenize
from pathlib import Path
data = Path("geertWildersTweets.txt").read_text().replace("\n","")
tweets = data.split("-") 

#print(tweets)

noiseLess_tweets = [];
for tweet in tweets:
    print(_lookup_words(tweet))
    # noiseLess_tweets.append(_remove_noise_str(
#word_tokenize(noiseLess_tweets, )

#N-grams as features

#def generate_ngrams(text, n):
#    words = text.split();
#    output = []
#   for i in range(len(words) - n + 1):
#        output.append(words[i:i+n])
#    return output

#print(generate_ngrams(doc1, 3))