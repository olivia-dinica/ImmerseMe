#!/usr/bin/env python
import re
import numpy as np
import csv
#breaks on the third line
mxm_data = open('cut_mxm_test.txt','r')


i=0
fnamecsv = 'test.csv'

with open(fnamecsv, 'w') as csvfile:
    for l in mxm_data:
        i+=1
        b=np.zeros(5000)
#        print "line number", i#read line by line
#        print "line", l
        all_l_split = l.split(',')
        id_split=all_l_split[0:2]
        l_split=all_l_split[2:]
#        print "only #",l_split

        for u in l_split: #read comma by comma
            n,c,freq = u.partition(':')
#            print "real",n,freq
            b[int(n)-1]=int(freq)
#            print "mine",int(n)-1,b[int(n)-1]
#        print b
        line=np.append(id_split,b)
        print i
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(line)

       
