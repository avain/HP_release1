#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import genfromtxt
import os
import csv
import collections
import numpy as np     ## vector and matrix operations
import scipy as sp     ## grab-bag of statistical and science tools
import matplotlib.pyplot as plt     ## matplotlib - plots
import pandas as pd     ## emulates R data frames
import statsmodels.api as sm     ## scikits.statsmodels - statistics library
import patsy    ## emulates R model formulas
import math
import sklearn as skl     ## scikits.learn - machine learning library
from sklearn import mixture as sklmix
MemberfileName="Members_Y1.csv"
global claimsAns
def svmtest():
    import numpy as np
    import pylab as pl
    from sklearn import svm
    
    
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    
    # fit the model
    clf = svm.NuSVC()
    clf.fit(X, Y)
    
    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    pl.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
              origin='lower', cmap=pl.cm.PuOr_r)
    contours = pl.contour(xx, yy, Z, levels=[0], linewidths=2,
                          linetypes='--')
    pl.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=pl.cm.Paired)
    pl.xticks(())
    pl.yticks(())
    pl.axis([-3, 3, -3, 3])
    pl.show()
    
def _svmprocess():
    """
    用svm 處理 numpy matrix 
    """
    np.random.shuffle(claimsAns[1:,:])
    """
    array(['MemberID', 'ProviderID', 'vendor', 'pcp', 'Year', 'specialty',
       'placesvc', 'paydelay', 'LengthOfStay', 'dsfs',
       'PrimaryConditionGroup', 'CharlsonIndex', 'AgeAtFirstClaim', 'Sex',
       'DayInHospital_Y2'], 
      dtype='|S25')

    """
    import numpy as np
    claimsAns=np.load("claims_join_member_ans_cat.npy")
    Y=[]
    X=[]
    for i in range(5000):
        Y.append(int(claimsAns[1+i,-1]))
        x=[int(claimsAns[1+i,-4]),int(claimsAns[1+i,-5])]#,int(claimsAns[1+i,-6])]  
        X.append(x)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    from sklearn import svm
    kernel='rbf'
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)
    
    
    clf.predict([[2., 2.]])
    #=============
    kernel='rbf'
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)
    
    # plot the line, the points, and the nearest vectors to the plane
    pl.figure(fignum, figsize=(4, 3))
    pl.clf()
    
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
               facecolors='none', zorder=10)
    pl.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=pl.cm.Paired)
    
    pl.axis('tight')
    x_min = 0
    x_max = 10
    y_min = 0
    y_max = 10
    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    
    pl.pcolormesh(XX, YY, Z > 0, cmap=pl.cm.Paired)
    pl.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])
    
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    
    pl.xticks(())
    pl.yticks(())

def _svmpreprocess():
    """
    array(['MemberID'0, 'ProviderID'1, 'vendor'2, 'pcp'3, 'Year'4, 'specialty'5,
       'placesvc'6, 'paydelay'7, 'LengthOfStay'8, 'dsfs'9,
       'PrimaryConditionGroup'10, 'CharlsonIndex'11, 'AgeAtFirstClaim'12, 'Sex'13], 
      dtype='|S25')

    """

    #convert Categorical to Integer
    catType=catHash()
    claimsAns=np.load("claims_join_member_ans.npy")
    for i in range(int(claimsAns.shape[1])):
        fieldName=claimsAns[0][i]
        if catType.has_key(fieldName):
            for j in range(int(claimsAns.shape[0])):
                if j==0:
                    pass
                else:
                    #sraw_input("waits")
                    #print fieldName
                    #print catType[fieldName]
                    #print i,j
                    #claimsAns[j][i]
                    #print catType[fieldName][claimsAns[j][i]]
                    claimsAns[j][i]=catType[fieldName][claimsAns[j][i]]
        return claimAns

    
def _loadcvsForFeautreAnalysis(fileName=MemberfileName):
    """
    return dict for basic feature analysis
    """
    f=open(fileName)
    data=f.read()
    f.close()
    items=data.split("\r\n")
    result={}
    indexCount={}
    for i, val in enumerate(items):
        fields=val.split(",")        
        if i==0:
            for j,f in enumerate(fields):
                indexCount[j]=f
                result[f]=[]            
        else:
            for j,f in enumerate(fields):                
                result[indexCount[j]].append(f)
    return result

def _getCategoricalfeatures(DataList,DictEnable=True):
    """
    http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
    """
    s=set(DataList)
    l=list(s)
    l.sort()
    if DictEnable:
        d=dict(enumerate(l))
        res = dict((v,k) for k,v in d.iteritems())
        return res
    else:
        return l
def MergeClaim_MemeberAndAns():
    claims=np.load("claims_join_member.npy")
    Y2Ans=loadcsv("DayInHospital_Y2.csv")
    DY2=np.empty([644707,1],dtype=str)
    claims=np.column_stack([claims,DY2])
    AnsDict={}
    for i in range(Y2Ans.shape[0]):
        AnsDict[Y2Ans[i][0]]=Y2Ans[i][1]
    claims[0,-1]='DayInHospital_Y2'
    for i in range(claims.shape[0]):
        if i==0:
            pass
        else:
            claims[i][-1]=AnsDict[claims[i][0]]
    np.save("claims_join_member_ans",claims)
    
    return claims
    



def MergeMaxrix():
    """
    Try it 
    http://stackoverflow.com/questions/877479/whats-the-simplest-way-to-extend-a-numpy-array-in-2-dimensions
    """
    member=loadcsv()
    claims=loadcsv("Claims_Y1.csv")
    insertA=np.empty([644707,2],dtype=str)
    claims=np.column_stack([claims,insertA])
    MemberDict={}
    for i in range(member.shape[0]):
        MemberDict[member[i][0]]=(member[i][1],member[i][2])
    claims[0,-2]='Sex'
    claims[0,-1]='AgeAtFirstClaim'
    for i in range(claims.shape[0]):
        if i==0:
            pass
        else:
            claims[i][-1]=MemberDict[claims[i][0]][0]
            claims[i][-2]=MemberDict[claims[i][0]][1]
    
    np.save("claims_join_member",claims)  #npy file
    #claims.tofile("claims_join_member.csv",sep=",") #csv file 沒有換行
    #np.savetxt("claims_join_member.csv",claims, delimiter=",")
    claims=np.load("claims_join_member.npy")

def convertField(x,mappingList):
    return mappingList.index(x)
    

def loadcsv(fileName=MemberfileName):    
    #persontype = np.dtype({'names':['MemberID','AgeAtFirstClaim','Sex'],'formats':['i','S10', 'S1']})
    #persontype = np.dtype({'names':['MemberID','AgeAtFirstClaim','Sex'],'formats':['i','i', 'i']})
    #persontype = [('MemberID','i'),('AgeAtFirstClaim','i'),('Sex','i')]
    #dtype=[('myint','i8'),('myfloat','f8'),('mystring','S5')]
    #my_data = genfromtxt(fileName, delimiter=',',dtype= persontype,skiprows=1)
    #d=_loadcvsForFeautreAnalysis(fileName)
    #ageList=_getCategoricalfeatures(d['AgeAtFirstClaim'],False)
    #sexList=_getCategoricalfeatures(d['Sex'],False)    
    #ageConv = lambda x: ageList.index(x)
    #sexConv = lambda x: sexList.index(x)
    #np.loadtxt(fileName, delimiter=',')
    #result = genfromtxt(fileName, delimiter=',',skiprows=1,converters = {1:ageConv,2:sexConv},dtype= persontype)
    #reader=csv.reader(open(fileName,"rb"),delimiter=',')
    #result=np.loadtxt(fileName, delimiter=',',skiprows=1,converters = {1:ageConv,2:sexConv},dtype= persontype)
    #result=np.loadtxt(fileName, delimiter=',',dtype= None)
    result=np.genfromtxt(fileName, delimiter=',',dtype= None)
    #x=list(reader)
    #result=np.array(x[1:])
    
    #print ageList
    #print sexList
    
    #for i in range(result.shape[0]):
    #    result[i][1]=ageList[result[i][1]]
    #    result[i][2]=sexList[result[i][2]]
    
    return result

def catHash():
    """
    charList
    Out[52]: {'0': 0, '1-2': 1, '3-4': 2, '5+': 3}
    dict(zip([1,2,3,4], [a,b,c,d]))
    """
    result={}
    result['LengthOfStay']={'': 0,
    '1 day': 1,
    '2 days': 2,
    '3 days': 3,
    '4 days': 4,
    '5 days': 5,
    '6 days': 6,
    '1- 2 weeks': 10,
    '2- 4 weeks': 20,
    '4- 8 weeks': 42,
    '8-12 weeks': 70,
    '12-26 weeks': 140,
    '26+ weeks': 168
    }
    result['AgeAtFirstClaim']={'0-9': 0,
    '10-19': 1,
    '20-29': 2,
    '30-39': 3,
    '40-49': 4,
    '50-59': 5,
    '60-69': 6,
    '70-79': 7,
    '80+': 8}
    result['Sex']={'F': 0, 'M': 1}
    
    #['paydelay':163,'vendor':3851, 'dsfs':12, 'placesvc':8, 'pcp':1012, 'MemberID':77290,
    #'CharlsonIndex':4, 'specialty':12, 'ProviderID':7825, 'PrimaryConditionGroup':45, 'Year':1, 'LengthOfStay':13]
    val=dict(enumerate(range(0,162)))
    result['paydelay']=dict((str(v),k) for v,k in val.iteritems())  #-1,0,1,...,161 共163筆
    result['paydelay']['']=-1
    
    result['dsfs']={'0- 1 month': 0,
    '1- 2 months': 1,
    '10-11 months': 2,
    '11-12 months': 3,
    '2- 3 months': 4,
    '3- 4 months': 5,
    '4- 5 months': 6,
    '5- 6 months': 7,
    '6- 7 months': 8,
    '7- 8 months': 9,
    '8- 9 months': 10,
    '9-10 months': 11}
    
    result['placesvc']={'Ambulance': 0,
    'Home': 1,
    'Independent Lab': 2,
    'Inpatient Hospital': 3,
    'Office': 4,
    'Other': 5,
    'Outpatient Hospital': 6,
    'Urgent Care': 7}
    
    result['CharlsonIndex']={'0': 0, '1-2': 1, '3-4': 2, '5+': 3}
    result['specialty']={'Anesthesiology': 0,
    'Diagnostic Imaging': 1,
    'Emergency': 2,
    'General Practice': 3,
    'Internal': 4,
    'Laboratory': 5,
    'Obstetrics and Gynecology': 6,
    'Other': 7,
    'Pathology': 8,
    'Pediatrics': 9,
    'Rehabilitation': 10,
    'Surgery': 11}
    result['PrimaryConditionGroup']={'AMI': 0,
    'APPCHOL': 1,
    'ARTHSPIN': 2,
    'CANCRA': 3,
    'CANCRB': 4,
    'CANCRM': 5,
    'CATAST': 6,
    'CHF': 7,
    'COPD': 8,
    'FLaELEC': 9,
    'FXDISLC': 10,
    'GIBLEED': 11,
    'GIOBSENT': 12,
    'GYNEC1': 13,
    'GYNECA': 14,
    'HEART2': 15,
    'HEART4': 16,
    'HEMTOL': 17,
    'HIPFX': 18,
    'INFEC4': 19,
    'LIVERDZ': 20,
    'METAB1': 21,
    'METAB3': 22,
    'MISCHRT': 23,
    'MISCL1': 24,
    'MISCL5': 25,
    'MSC2a3': 26,
    'NEUMENT': 27,
    'ODaBNCA': 28,
    'PERINTL': 29,
    'PERVALV': 30,
    'PNCRDZ': 31,
    'PNEUM': 32,
    'PRGNCY': 33,
    'RENAL1': 34,
    'RENAL2': 35,
    'RENAL3': 36,
    'RESPR4': 37,
    'ROAMI': 38,
    'SEIZURE': 39,
    'SEPSIS': 40,
    'SKNAUT': 41,
    'STROKE': 42,
    'TRAUMA': 43,
    'UTI': 44}
    return result   







def showHistogram(data):
    data=list(data)
    correlation = [(i, data.count(i)) for i in set(data)]
    correlation.sort(key=lambda x: x[1])    
    labels, values = zip(*correlation)
    indexes = np.arange(len(correlation))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


def _countFreq(items):
    """
    print(counter)
    # Counter({1: 4, 2: 4, 3: 2, 5: 2, 4: 1})
    print(counter.values())
    # [4, 4, 2, 1, 2]
    print(counter.keys())
    # [1, 2, 3, 4, 5]
    print(counter.most_common(3))
    # [(1, 4), (2, 4), (3, 2)]
    """
    counter=collections.Counter(items)
    return counter
    
def freqDist():
    """
    計算頻率分佈
    """
    import numpy as np
    import analysis as a 
    claims=np.load("claims_join_member.npy")
    nameList=list(claims[0])
    for i in range(3,14):
        print "===========",nameList[i],"==================="
        print a._countFreq(claims[1:,i])
        print "=========================================================="

def test():
    """
    1.計算feature與住院的關聯
    2.住院與不住院的差異?
    3.今年住院跟明年有關嗎?
    """
    #filter
    #c=claims[(claims[:,12]=="30-39") | (claims[:,12]=="40-49")]
    l=list(set(claims[:12]))
    l.sort()
    for i in l:
        c=claims[(claims[:,12]==i)]
    print np.corrcoef(a,b)
    
    
def summary(filePath="Claims_Y1.csv"):
    fileName, fileExtension = os.path.splitext(filePath)
    if fileExtension==".npy":
        claimsAns=np.load(filePath)    
        d=pd.DataFrame(claimsAns)
    elif fileExtension==".csv":
        d=pd.read_csv(filePath)
    d.describe()
    return d
def RMSLE_Release1():
    dihY2=pd.read_csv("DayInHospital_Y2.csv",index_col='memberid')
    allZero_RMSLE=math.sqrt(sum(dihY2['DaysInHospital_Y2'].apply(math.log1p).pow(2))/dihY2.shape[0])
    print allZero_RMSLE
    #dihY2['ans'] = pd.Series(dihY2['DaysInHospital_Y2'], index=dihY2.index)
    dihY2['ans'] = pd.Series([0]*dihY2.shape[0], index=dihY2.index)
    #print dihY2
    diff=dihY2['DaysInHospital_Y2'].apply(math.log1p)-dihY2['ans'].apply(math.log1p)    
    #print diff
    diff=diff.pow(2)
    #print diff
    print math.sqrt(sum(diff)/dihY2.shape[0])
    #print math.sqrt(sum(diff)/dihY2.shape[0])
    return dihY2

def RMSLE_Release3():
    dihY2=pd.read_csv("DayInHospital_Y2.csv",index_col='MemberID')
    allZero_RMSLE=math.sqrt(sum(dihY2['DaysInHospital_Y2'].apply(math.log1p).pow(2))/dihY2.shape[0])
    print allZero_RMSLE
    #dihY2['ans'] = pd.Series(dihY2['DaysInHospital_Y2'], index=dihY2.index)
    dihY2['ans'] = pd.Series([0]*shape[0], index=dihY2.index)
    #print dihY2
    diff=dihY2['DaysInHospital'].apply(math.log1p)-dihY2['ans'].apply(math.log1p)    
    diff=diff.pow(2)
    print math.sqrt(sum(diff)/dihY2.shape[0])
    #math.sqrt(sum(/dihY2.shape[0])
    return dihY2
def calimsPlot():
    """
    ipython -pylab
    """
    claims=pd.read_csv("Claims_Y1.csv",index_col='MemberID')
    for i in claims.columns:        
        t=claims[i].value_counts()
        if t.count()>30:
            t.plot(title=i)
        else:
            t.plot(title=i,kind='bar',yticks=t)
        raw_input("wait...")
        plt.clf()
            
def Test():
    from sklearn import cross_validation
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    m=pd.load("featuresSet")
    result={}
    for i in range(15,299):
        X=m[[m.columns[i]]]
        print m.columns[i]
        Y=m["DaysInHospital_Y2"]
        X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,Y,test_size=0.95,random_state=0)
        clf = RandomForestClassifier(n_estimators=100)
        #clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        
        #clf = clf.fit(X_train, y_train)
        score=clf.score(X_test, y_test)   
        result[m.columns[i]]=score
        print m.columns[i],score
    return result
    



def Merge():
    dihY2=pd.read_csv("DayInHospital_Y2.csv")
    dihY2.rename(columns={'memberid':'MemberID'},inplace=True)
    claims=pd.read_csv("Claims_Y1.csv")        
    m=pd.merge(claims,dihY2,on='MemberID')
    member=pd.read_csv("Members_Y1.csv")
    m=pd.merge(m,member,on='MemberID')    
    return m
    #expend features
    """
    Index([u'MemberID', u'ProviderID', u'vendor', u'pcp', u'Year', u'specialty', u'placesvc', u'paydelay', u'LengthOfStay', u'dsfs',
    u'PrimaryConditionGroup', u'CharlsonIndex', u'DaysInHospital_Y2', u'sex', u'AgeAtFirstClaim'], dtype='object')
    MemberID (77289,)
    ProviderID (7825,)
    vendor (3851,)
    pcp (1012,)
    Year (1,)
    specialty (12,)
    placesvc (8,)
    paydelay (163,)
    LengthOfStay (13,)
    dsfs (12,)
    PrimaryConditionGroup (45,)
    CharlsonIndex (4,)
    DaysInHospital_Y2 (16,)
    sex (2,)
    AgeAtFirstClaim (9,)
    
    new features:15-299
    """
    for i in m.columns:
        
        if i=="MemberID" or i=="ProviderID" or i=="vendor" or i=="pcp" or i=="Year":
            continue
        
        else:
            categories=m[i].unique()
            for j in categories:
                #print j
                print m.shape
                fieldName=i+str(j)
                fieldName=fieldName.replace(" ","_")
                fieldName=fieldName.replace(".0","")                            
                m[fieldName]=pd.Series((m[i]==j)+0, index=m.index)
                #print m[fieldName]
                #raw_input("wait for me")
            
    return m
    
    
    
    

