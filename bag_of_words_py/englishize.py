#!/usr/bin/env python

from nltk import pos_tag,word_tokenize
from nltk.corpus import wordnet
with open ("header_w_spaces", "r") as f:
    header=f.read()
#print header
englishwords = []
tokenized_set=word_tokenize(header)
#print tokenized_set

for word in tokenized_set:
    print word
    if wordnet.synsets(word):
        englishwords.append(word)
