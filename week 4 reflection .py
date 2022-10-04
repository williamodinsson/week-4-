#!/usr/bin/env python
# coding: utf-8

# In[1]:


target = 22367 # Ebook number
import gutenbergpy.textget
raw = gutenbergpy.textget.get_text_by_id(target)
book= gutenbergpy.textget.strip_headers(raw)
count = 50


# In[4]:


s = book.decode("utf-8") # get a string from the byte sequence
startmarker = '1917'
endmarker = 'Tochter als erste sich erhob und ihren jungen Körper dehnte.'
startPosition = s.index(startmarker) + len(startmarker)
endPosition = s.index(endmarker)+len(endmarker)
content = s[startPosition:endPosition]
print(content[:count]) # start
print(content[-count:]) # end


# In[5]:


import re
# extract Roman Numerals and replace them with 'ROMAN' by using Regular Expression
text = re.sub(r'(?=\b[MCDXLVI]{1,6}\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})', 'ROMAN', content)
#print(text)


# In[6]:


# count the number of sections
sections = text.count('ROMAN')

# removing the sections headers
print('No digits:', sections)
clean = re.compile(r'\s+') # also combine any kind of repeated whitespace into a single space
ok = clean.sub(' ', text)
print('Cleaned:', ok[:count])
potential = ok.split('ROMAN.')
stripped = [ candidate.strip().lstrip() for candidate in potential ] # remove leading and trailing space 
real = [ s for s in stripped if len(s) > 0 ] # keep only the ones with content
print(len(real), 'real chapters')
print(real[0][:count])


# In[7]:


# convert list to a string
real_text = ' '
for x in real:
    real_text += ' ' + x


# In[8]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
skip = stopwords.words('german')


# In[9]:


#Removal of stopwords
wordsFiltered = []
for w in real_text.split():
    if w not in skip:
        wordsFiltered.append(w)


# In[10]:


wordsFiltered[1:10]


# In[11]:


#Cleaned corpus after stopwords are removed
real_new = ' '
for x in wordsFiltered:
    real_new += ' ' + x


# In[12]:


real_new[1:100]

1. How does the precision of an n-gram tagger behave as you increase the value of n from
one to k where k > 3 is the value of your choice (depending on the computing resources
you have at hand).
You are free to choose your own corpus (it does not have to be brown like in the examples)
# #### Without punctuations

# In[13]:


#Cleaned corpus after stopwords & punktuations are removed 
new_string = re.sub(r'[^\w\s]', '', real_new)
new_string[1:97]


# In[14]:


nltk.download('averaged_perceptron_tagger')


# In[15]:


tokens = nltk.word_tokenize(new_string)
tags = nltk.pos_tag(tokens)
tags [4:8]


# #### The result is completely inaccurate. The 'Averaged Perceptron Tagger' is not created for German.

# In[52]:


text = ['Als Gregor Samsa Morgens unruhigen Träumen erwachte fand Bett ungeheueren Ungeziefer verwandelt Er lag panzerartig harten Rücken sah Kopf wenig hob gewölbten braunen bogenförmigen Versteifungen geteilten Bauch Höhe Bettdecke gänzlichen Niedergleiten bereit kaum erhalten konnte Seine vielen Vergleich sonstigen Umfang kläglich dünnen Beine flimmerten hilflos Augen']


# In[307]:


import os, glob, codecs

def installStanfordTag():
    if not os.path.exists('/Users/vilhjalmur-odinsson/Downloads/stanford-postagger-full-2013-06-20'):
        os.system('wget http://nlp.stanford.edu/software/stanford-postagger-full-2013-06-20.zip')
        os.system('unzip stanford-postagger-full-2013-06-20.zip')
    return

def tag(infile):
    cmd = "./stanford-postagger.sh "+models[m]+" "+infile
    tagout = os.popen(cmd).readlines()
    return [i.strip() for i in tagout]

def taglinebyline(sents):
    tagged = []
    for ss in sents:
        os.popen("echo '''"+ss+"''' > stanfordtemp.txt")
        tagged.append(tag('stanfordtemp.txt'))
    return tagged

installStanfordTag()
stagdir = '/Users/vilhjalmur-odinsson/Downloads/stanford-postagger-full-2013-06-20'
models = {'fast':'models/german-fast.tagger',
          'dewac':'models/german-dewac.tagger',
          'hgc':'models/german-hgc.tagger'}
os.chdir(stagdir)
#print os.getcwd()


m = 'fast' 


tagged_sents = taglinebyline(text) # Call the stanford tagger

for sent in tagged_sents:
     print (sent)


# #### The Standford POS Tagger works very well

# In[95]:


from HanTa import HanoverTagger as ht
tagger = ht.HanoverTagger('morphmodel_ger.pgz')
words = nltk.word_tokenize(new_string)


# In[97]:


tagger.tag_sent(words)[0:3]


# In[101]:


import numpy as np
a = tagger.tag_sent(words)
b = np.array(a)
d = b[:,-2:].tolist()
sss = [tuple(l) for l in d]


# In[100]:


size = int(len(tagger.tag_sent(words))*0.8)

train = [sss[:size]]

test = [sss[size:]]


# In[173]:


from nltk.tag import DefaultTagger 
from nltk.tag import UnigramTagger 
from nltk.tag import BigramTagger 
from nltk.tag import TrigramTagger 


# In[179]:


t1 = nltk.UnigramTagger(train)
print(t1.accuracy(test))


# In[156]:


bigrams = ngrams(words,2)
bi_cluster = []
for i in bigrams:
    bi_cluster.append(i)


# In[157]:


bi_cluster[0:3]


# In[158]:


bi = []
for x in bi_cluster:
    #print(x)
    w = ','.join(x)
    bi.append(w)


# In[159]:


a1 = tagger.tag_sent(bi)
b1 = np.array(a1)
d1 = b1[:,-2:].tolist()
sss1 = [tuple(l) for l in d1]


# In[232]:


sss1


# In[160]:


size1 = int(len(tagger.tag_sent(bi))*0.8)

train1 = [sss1[:size1]]

test1 = [sss1[size1:]]


# In[429]:


bi


# In[180]:


t2 = nltk.BigramTagger(train1)
print(t2.accuracy(test1))


# In[181]:


#Backoff to UnigramTagger
t2 = nltk.UnigramTagger(train1)
print(t2.accuracy(test1))


# In[163]:


trigrams = ngrams(tokens,3)
tri_cluster = []
for i in trigrams:
    tri_cluster.append(i)


# In[165]:


tri = []
for x in tri_cluster:
    #print(x)
    w = ','.join(x)
    tri.append(w)


# In[167]:


a2 = tagger.tag_sent(tri)
b2 = np.array(a2)
d2 = b2[:,-2:].tolist()
sss2 = [tuple(l) for l in d2]


# In[168]:


size2 = int(len(tagger.tag_sent(tri))*0.8)

train2 = [sss2[:size2]]

test2 = [sss2[size2:]]


# In[169]:


t3 = nltk.TrigramTagger(train2)
print(t3.accuracy(test2))


# In[170]:


#Backoff to BigramTagger
t3 = nltk.BigramTagger(train2)
print(t3.accuracy(test2))


# In[171]:


#Backoff to UnigramTagger
t3 = nltk.UnigramTagger(train2)
print(t3.accuracy(test2))


# #### As n increases, the accuracy drops DRAMATICALLY
2. What is the effect on that precision when sentence breaks are taken into account versus
when they are ignored? (See the section Tagging Across Sentence Boundaries in the
Python textbook, Chapter 5).
# In[187]:


tokens = nltk.word_tokenize(real_new)


# In[189]:


aa = tagger.tag_sent(tokens)
bb = np.array(aa)
dd = bb[:,-2:].tolist()
ssss = [tuple(l) for l in dd]


# In[190]:


sizes = int(len(tagger.tag_sent(tokens))*0.8)

trains = [ssss[:sizes]]

tests = [ssss[sizes:]]


# In[191]:


T1 = nltk.UnigramTagger(trains)
print(T1.accuracy(tests))


# In[198]:


bigrams1 = ngrams(tokens,2)
bi_cluster1 = []
for i in bigrams1:
    bi_cluster1.append(i)


# In[199]:


bis = []
for x in bi_cluster1:
    #print(x)
    w = ','.join(x)
    bis.append(w)


# In[200]:


aa1 = tagger.tag_sent(bis)
bb1 = np.array(aa1)
dd1 = bb1[:,-2:].tolist()
ssss1 = [tuple(l) for l in dd1]


# In[201]:


sizes1 = int(len(tagger.tag_sent(tokens))*0.8)

trains1 = [ssss1[:sizes1]]

tests1 = [ssss1[sizes1:]]


# In[202]:


T2 = nltk.BigramTagger(trains1)
print(T2.accuracy(tests1))


# In[229]:


#Backoff to Unigram
T2 = nltk.UnigramTagger(trains1)
print(T2.accuracy(tests1))


# In[206]:


trigrams1 = ngrams(tokens,3)
tri_cluster1 = []
for i in trigrams1:
    tri_cluster1.append(i)


# In[207]:


tris = []
for x in tri_cluster1:
    #print(x)
    w = ','.join(x)
    tris.append(w)


# In[208]:


aa2 = tagger.tag_sent(tris)
bb2 = np.array(aa2)
dd2 = bb2[:,-2:].tolist()
ssss2 = [tuple(l) for l in dd2]


# In[210]:


sizes2 = int(len(tagger.tag_sent(tokens))*0.8)

trains2 = [ssss2[:sizes2]]

tests2 = [ssss2[sizes2:]]


# In[226]:


T3 = nltk.TrigramTagger(trains2)
print(T3.accuracy(tests2))


# In[230]:


#Backoff to BigramTagger
T3_1 = nltk.BigramTagger(trains2)
print(T3_1.accuracy(tests2))


# In[231]:


#Backoff to UnigramTagger
T3_2 = nltk.UnigramTagger(trains2)
print(T3_2.accuracy(tests2))


# In[262]:


text


# In[278]:


real_text


# In[269]:


s = book.decode("utf-8") # get a string from the byte sequence
startmarker = '\n\n\n\n\nI.'
endmarker = 'Die Tür wurde noch mit dem\nStock zugeschlagen, dann war es endlich still.'
startPosition = s.index(startmarker) + len(startmarker)
endPosition = s.index(endmarker)+len(endmarker)
corp1 = s[startPosition:endPosition]


# In[270]:


startmarker = '\n\n\n\n\nII.'
endmarker = 'die Hände an des Vaters Hinterkopf um Schonung von\nGregors Leben bat.'
startPosition = s.index(startmarker) + len(startmarker)
endPosition = s.index(endmarker)+len(endmarker)
corp2 = s[startPosition:endPosition]


# In[271]:


startmarker = '\n\n\n\n\nIII.'
endmarker = 'Tochter als erste sich erhob und ihren jungen Körper dehnte.'
startPosition = s.index(startmarker) + len(startmarker)
endPosition = s.index(endmarker)+len(endmarker)
corp3 = s[startPosition:endPosition]


# In[385]:


clean1 = re.compile(r'\s+') # also combine any kind of repeated whitespace into a single space
ok1 = clean1.sub(' ', corp1)


# In[386]:


clean2 = re.compile(r'\s+') 
ok2 = clean2.sub(' ', corp2)


# In[387]:


clean3 = re.compile(r'\s+') 
ok3 = clean3.sub(' ', corp3)


# In[388]:


#Removal of stopwords
real1 = []
for w in ok1.split():
    if w not in skip:
        real1.append(w)


# In[389]:


real2 = []
for w in ok2.split():
    if w not in skip:
        real2.append(w)


# In[390]:


real3 = []
for w in ok3.split():
    if w not in skip:
        real3.append(w)


# In[396]:


#Cleaned corpus after stopwords are removed
r1= ' '
for x in real1:
    r1 += ' ' + x


# In[398]:


r2= ' '
for x in real2:
    r2 += ' ' + x


# In[400]:


r3= ' '
for x in real3:
    r3 += ' ' + x


# In[402]:


from nltk.tokenize import sent_tokenize
rr1 = list(sent_tokenize(r1))
rr2 = list(sent_tokenize(r2))
rr3 = list(sent_tokenize(r3))


# In[403]:


L1 = []
L2=[]
L3=[]

# Creating list of list format
for elem in rr1:
    temp = elem.split(', ')
    L1.append((temp))

for elem in rr2:
    temp = elem.split(', ')
    L2.append((temp))
    
for elem in rr3:
    temp = elem.split(', ')
    L3.append((temp))

# Final list
res1 = []
res2 = []
res3 =[]

for elem in L1:
    temp = []
    for e in elem:
        temp.append(e)
    res1.append(temp)
    
for elem in L2:
    temp = []
    for e in elem:
        temp.append(e)
    res2.append(temp)
    
for elem in L3:
    temp = []
    for e in elem:
        temp.append(e)
    res3.append(temp)

# printing
# print("The list of lists:\n",res1)
# print("The list of lists:\n",res2)
# print("The list of lists:\n",res3)


# In[409]:


import time


for i in res1:
    q = taglinebyline(i)
    print (q)

start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[407]:


for i in res2:
    q = taglinebyline(i)
    print (q)
    
start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[408]:


for i in res3:
    q = taglinebyline(i)
    print (q)
    
start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[294]:


from flair.data import Sentence
from flair.models import SequenceTagger


# In[411]:




#tagger = SequenceTagger.load('pos-multi')
from flair.tokenization import SegtokSentenceSplitter

splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(r1)

# predict tags for sentences

tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
    
    
start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[412]:


# use splitter to split text into list of sentences
sentences2 = splitter.split(r2)

# predict tags for sentences

tagger.predict(sentences2)

# iterate through sentences and print predicted labels
for sentence in sentences2:
    print(sentence)
    
    
start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[461]:


# use splitter to split text into list of sentences
sentences3 = splitter.split(r3)

# predict tags for sentences

tagger.predict(sentences3)

# iterate through sentences and print predicted labels
for sentence in sentences3:
    print(sentence)
    
    
start = time.process_time()  
print('\n\n\n'+ str(time.process_time() - start))


# In[ ]:




