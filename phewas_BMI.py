import csv
import math
from sets import Set
from random import randint
import pandas as pd


def removeVERecord(sourcefile, mapfile, outputfile):

    ESet = Set()
    VSet = Set()
    VEMappingSet = Set()
    regularset = Set()
    intersectSet = Set()

    pfile = open(mapfile, 'rb')
    reader = csv.reader(pfile)
    next(reader) #skip first row (header)
    for row in reader:
    	if 'E' in row[0]: #any phewas with a E code mapping, remove it
	    ESet.add(row[2]) 
	elif 'V' in row[0]:
	    if row[2] not in regularset: 
		VSet.add(row[2])
	    else:
		intersectSet.add((row[0],row[2]))
	else :
	    regularset.add(row[2])
    pfile.close()

    VEMappingSet = ESet.union(VSet) 
    print "E set size", len(ESet)
    print "V set size", len(VSet)
    print "VE Mapping set size", len(VEMappingSet)
    print "Intersect set size", len(intersectSet)
    print "Intersect set:"
    print intersectSet


    print "start writing ", outputfile
    ofile = open(outputfile, 'wb')
    writer = csv.writer(ofile,delimiter=',',)

    pfile = open(sourcefile, 'rb')
    reader = csv.reader(pfile)
    writer.writerow(next(reader))
    for row in reader:
    	if row[2] != "": #has a phewas code
    		pid,pname= row[2].split('-',1)
    		if pid not in VEMappingSet:
		    writer.writerow(row)

    pfile.close()
    ofile.close()
    print "writing finished!"
 
#read the file, return subs, phws
def readFile(sourcefile):
    subs = {} 
    phws = Set() 

    pfile = open(sourcefile, 'rb')
    reader = csv.reader(pfile)
    next(reader)
    for row in reader:
    	sid = row[0]	
    	if row[2] != "": #has a phewas code
    		pid,pname= row[2].split('-',1)
    		phws.add(pid)
    		if sid not in subs:
    			subs[sid] = Set() 
    		subs[sid].add(pid)		
    pfile.close()
    return subs,phws


def rollupPhewas(subs):
    rollsubs = {} 
    rollphws = Set()
    for sid in subs:
	rollsubs[sid] = Set()
	for pid in subs[sid]:
	    intp = int(math.floor(float(pid)))
	    sp = str(intp)
	    rollsubs[sid].add(sp)
	    rollphws.add(sp)
    return rollsubs,rollphws

def addLabel(subs,demogfile,genMethod=''):
    labels = {}
    lfile = open(demogfile, 'rb')
    reader = csv.reader(lfile)
    next(reader)
    for row in reader:
    	sid = row[0]
	if sid in subs:
	   if genMethod == 'gender':
    	   	labels[sid] = (1 if row[1] == 'C46110' else 0)
    
    	   elif genMethod == 'age':
    	   	try:
    	   		labels[sid] = (1 if int(row[3]) >= 1990 else 0)
    	   	except:
    	   		pass
    	   else: #default,random
    	   	labels[sid] = (1 if randint(0,9) == 0 else 0)
    
    lfile.close()
    return labels 

def addBMILabel(subs,demogfile,genMethod=''):
    labels = {}
    lfile = open(demogfile, 'rb')
    reader = csv.reader(lfile)
    next(reader)
    for row in reader:
    	sid = row[0]
	if sid in subs:
	    labels[sid] = row[8]
	    #print labels[sid]

    lfile.close()
    return labels 


#output matrix:sub_id phw1 phw2 phw3 ... label
#	       id1    1/0  1/0  1/0      1/0
#	       id2    1/0  1/0  1/0      1/0
#	       ...

def output_csv(subs,phws,labels,outputfile):

    #phws = sorted(phws)
    phws = sorted(phws, key=lambda x : float(x))
    
    print "Writing subjects..."
    wsfile = open(outputfile,'wb')
    writer = csv.writer(wsfile,delimiter=',',)
    header = ['Subject ID'] + list(phws) + ['Label']
    writer.writerow(header)
    for sid  in subs: 
	#print subs[sid]
    	ftrs = []
    	ftrs.append(sid)
    	for pid in phws:
    		ftrs.append(1 if pid in subs[sid] else 0)
	ftrs.append(labels[sid])
    	writer.writerow( ftrs )
    wsfile.close()
    print "finished!"
    return 

   		
def mergePhewas():
    return 
    

def getPhewas(sourcefile, VEMappingSet): 

    #subjects K = subjectID, V = [phwsId1, phwsId2...]} 
    subs = {} 
    phws = Set() 
    
    pfile = open(sourcefile, 'rb')
    reader = csv.reader(pfile)
    next(reader)
    for row in reader:
    	sid = int(row[0])	
    	if row[2] != "": #has a phewas code
    		pid,pname= row[2].split('-',1)
    		pid = float(pid)
    		if pid not in VEMappingSet:
    			phws.add(pid)
    			if sid not in subs:
    				subs[sid] = Set() 
    			subs[sid].add(pid)		
    		
    pfile.close()

    total =  sum( len(e) for e in subs)
    print "total pairs", total

    phws = sorted(phws)

from sklearn import linear_model
from sklearn.linear_model import lasso_path, enet_path
from itertools import cycle
import math
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 




df = pd.read_csv("data/ml_rollup_labeled_Phewas.csv")

X = df.drop(['Subject ID','Label'], axis = 1)
X = X.T 

subjectids = df['Subject ID']
Y = df['Label']

print (Y)
print (subjectids)

ly = list(Y)
lsids = list(subjectids)

tdf = pd.read_csv("data/tdml_rollup_labeled_Phewas.csv")
tX = tdf.drop(['Subject ID','Label'], axis = 1)
tX = tX.T 
tY = tdf['Label']


clf = linear_model.Lasso(alpha=1)
clf.fit( X.T, Y)
lcoefs = list(clf.coef_)
print len(lcoefs)

#clf.predict(tX.T)
print clf.score(tX.T,tY)

#print(clf.coef_)
#print(clf.intercept_)

rlf = linear_model.Ridge(alpha=1)
rlf.fit(X.T,Y)
rcoefs = list(rlf.coef_)
print len(rcoefs)
print rlf.score(tX.T,tY)


elf = linear_model.ElasticNet(l1_ratio=0.5)
elf.fit(X.T,Y)
ecoefs = list(elf.coef_)
print len(ecoefs)
print elf.score(tX.T,tY)

xaxis = np.arange(len(lcoefs))
plt.bar(xaxis, lcoefs,align='center',alpha=0.5,width=0.01)
plt.ylabel('Weight')
plt.show()

plt.bar(xaxis, rcoefs,align='center',alpha=0.5,width=0.01)
plt.ylabel('Weight')
plt.show()

plt.bar(xaxis, ecoefs,align='center',alpha=0.5,width=0.01)
plt.ylabel('Weight')
plt.show()



eps = .05  # the smaller it is the longer is the path
print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X.T, Y, eps, fit_intercept=False)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X.T, Y, eps=eps, l1_ratio=0.5, fit_intercept=False)

# Display results

plt.figure(4)
ax = plt.gca()

colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')

plt.show()

