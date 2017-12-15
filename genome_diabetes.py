import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import lasso_path, enet_path
import os

#Build X matrix: find the common subjects in genome and phenome, extract corresponding rows from genome dataset.
#Build Y (diabetes) target: first sort table based on subject ID, then find common index, extract diabetes related columns. Then merge them.


gensub = pd.read_csv('data/geno_data/combined_20150601.eMERGE.SNP.012.indv',header=None)
gensub.columns = ['SubID']
print gensub

phensub = pd.read_csv('data/pheno_data/subID',header=None)
phensub.columns = ['SubID']

subs = np.intersect1d(gensub, phensub)
print subs
print "total subjects"
print len(subs)

def gen_genMat():
    gsubidx = []
    for idx in subs:
        a = np.where(gensub==idx)
        gsubidx.append(a[0][0])
    print gsubidx
    ofile = open('data/geno_data/combined_20150601.eMERGE.SNP.012','r')
    wfile = open('data/gen_mat.csv','w')
    reader = csv.reader(ofile)
    writer = csv.writer(wfile)
    line = 0
    wline = 0
    for row in reader:
        if line in gsubidx:
            writer.writerow(row)
            wline += 1
        line += 1
        if line%100 == 0:
            print line

    print wline
    ofile.close()
    wfile.close()



def gen_target():
    df = pd.read_csv('data/pheno_data/SPHINX_phws_unrolled_idv.csv')
    df = df[['Subject ID','249','250','250.1','250.11','250.12','250.13','250.14','250.15','250.2','250.21','250.22','250.23','250.24','250.25','250.4','250.41','250.42','250.5','250.6','250.7']]
    df.info()
    df = df.sort_values('Subject ID')
    df.to_csv('data/tmp.csv',index=False)

    ofile = open('data/tmp.csv','r')
    wfile = open('data/gen_target.csv','w')
    reader = csv.reader(ofile)
    writer = csv.writer(wfile)
    alist = []
    line = 0
    wline = 0
    next(reader)
    for row in reader:
        for idx in subs:
            if int(row[0]) == idx:
                wline += 1
                if '1' in row[1:]:
                    writer.writerow([1])
                else:
                    writer.writerow([0])
                break
        line+=1
    print "total write"
    print wline
    ofile.close()
    wfile.close()
    os.remove('data/tmp.csv')





#gen_genMat()
gen_target()


from sklearn import linear_model
from sklearn.linear_model import lasso_path, enet_path
from itertools import cycle
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



reader = csv.reader(open('data/gen_mat.csv','r'),delimiter='\t')
line = 1
X = []
for row in reader:
    alst = row[1:]
    X.append( map(lambda x: float(x),alst) )
    print line
    line+=1

X = np.asmatrix(X)

reader = csv.reader(open('data/gen_target.csv','r'),delimiter='\t')
line = 1
Y = []
for row in reader:
    Y.append( float(row[0]))
    print line
    line+=1

print len(Y)

def lasso():
    clf = linear_model.Lasso(alpha=0.0001)
    clf.fit(X, Y)
    lcoefs = list(clf.coef_)
    print len(lcoefs)

    tfile = open('data/lassocoef.txt','w')
    for item in lcoefs:
        tfile.write("%s\n"%item)
    tfile.close()

    print "accuracy"
    print clf.score(X,Y)
#lasso()


def ridge():
    rlf = linear_model.Ridge(alpha=1)
    rlf.fit(X,Y)
    rcoefs = list(rlf.coef_)
    tfile = open('data/ridgecoef.txt','w')
    for item in rcoefs:
        tfile.write("%s\n"%item)
    tfile.close()
    print len(rcoefs)
    print "accuracy"
    print rlf.score(X,Y)


#ridge()


def elf():
    elf = linear_model.ElasticNet(l1_ratio=0.5,alpha=0.0001)
    elf.fit(X, Y)
    ecoefs = list(elf.coef_)
    tfile = open('data/enetcoef.txt','w')
    for item in ecoefs:
        tfile.write("%s\n"%item)
    tfile.close()
    print len(ecoefs)
    print elf.score(X, Y)

elf()


import matplotlib.pyplot as plt
import csv


def plot(sfile):
    x = []
    counter = 0
    reader = csv.reader(open(sfile,'r'),delimiter='\t')
    for row in reader:
        x.append( float(row[0]))
        if float(row[0]) != 0:
            counter +=1

    print counter
    plt.hist(x, normed=True, bins=10)
    plt.ylabel('counts')
    plt.show()

#plot('data/lassocoef0.01.txt')
#plot('data/lassocoef.txt')
#plot('data/ridgecoef.txt')
plot('data/enetcoef.txt')


