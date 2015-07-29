#!/usr/bin/env python

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

writef2=open("mood_lyrics_slow.csv", "w")
mood_slow=[""]*26977
#mood_slow=[""]*10
c=-1
with open("freq_to_string", "r") as f:
    for line in f:
        c+=1
        print c
        nba = TextBlob(line, analyzer=NaiveBayesAnalyzer())
        mood_slow[c]=str(nba.sentiment[1])+","+str(nba.sentiment[2])
        #print mood_slow[c]
        writef2.write("%s\n" % mood_slow[c])

writef2.close()
