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
    dihY2['ans'] = pd.Series([0]*dihY2.shape[0], index=dihY2.index)
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
            
def Featurescorr():
    """
    DaysInHospital_Y215 = 0.679839894557
    DaysInHospital_Y20  = -0.679463423154
    DaysInHospital_Y28  = 0.239186703963
    DaysInHospital_Y210 = 0.237311461682
    DaysInHospital_Y214 = 0.235583859674    
    DaysInHospital_Y212 = 0.236856544715
    DaysInHospital_Y26 = 0.206533784348
    """
    from sklearn import cross_validation
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    m=pd.load("featuresSet")
    result={}
    for i in range(15,299):
        fCorr=m[m.columns[i]].corr(m["DaysInHospital_Y2"])
        #if ((fCorr>0.2) or (fCorr<-0.2)):
        print m.columns[i],"=",fCorr
        result[m.columns[i]]=fCorr
        
    return result    

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
    
def TestDays():
    """
    1.DaysInHospital_Y20 = -0.679463423154 
    2.DaysInHospital_Y215 = 0.679839894557
    test_size=0.95
    1->0.849336540016
    1+2->0.86902890096
    all->0.906261684227
    all+16->0.906261684227
    all+paydelay76->0.906261684227
    
    test_size=0.9
    all->0.906158873286
    
    """
    from sklearn import cross_validation
    from sklearn import svm
    import itertools
    from sklearn.ensemble import RandomForestClassifier
    #itertools.permutations(["DaysInHospital_Y20","DaysInHospital_Y215","DaysInHospital_Y212","DaysInHospital_Y26","DaysInHospital_Y28","DaysInHospital_Y210", "DaysInHospital_Y214"])
    features=[
     "DaysInHospital_Y26",
     "DaysInHospital_Y212",
     "DaysInHospital_Y214",
     "DaysInHospital_Y210",
     "DaysInHospital_Y28",
     "DaysInHospital_Y20",
     "DaysInHospital_Y215" 
    ]
    combs = []
    for i in xrange(1, len(features)+1):
        #combs.append(i)
        els = [list(x) for x in itertools.combinations(features, i)]
        combs.extend(els)
    #return combs    
    m=pd.load("featuresSet")
    result={}    
    for i in combs:
        X=m[i]
        Y=m["DaysInHospital_Y2"]
        X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,Y,test_size=0.95,random_state=0)
        #clf = RandomForestClassifier(n_estimators=100)
        #clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        
        #clf = clf.fit(X_train, y_train)
        score=clf.score(X_test, y_test)   
        print i,"=>",score
        result[str(i)]=score
    return result


def Merge():
    dihY2   =pd.read_csv("DayInHospital_Y2.csv")
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
    
    
"""
{'AgeAtFirstClaim0-9': -0.051990535358995144,
 'AgeAtFirstClaim10-19': -0.053120585214221083,
 'AgeAtFirstClaim20-29': -0.020055519475893367,
 'AgeAtFirstClaim30-39': -0.034121723301183973,
 'AgeAtFirstClaim40-49': -0.070829706576915008,
 'AgeAtFirstClaim50-59': -0.051728956041176576,
 'AgeAtFirstClaim60-69': -0.012948134422449459,
 'AgeAtFirstClaim70-79': 0.064311863282610862,
 'AgeAtFirstClaim80+': 0.13442560590738883,
 'CharlsonIndex0': -0.13824233051485832,
 'CharlsonIndex1-2': 0.12582855508262986,
 'CharlsonIndex3-4': 0.047233068655335698,
 'CharlsonIndex5+': 0.010936434562424569,
 'DaysInHospital_Y20': -0.67946342315363084,
 'DaysInHospital_Y21': -0.0087339388361676908,
 'DaysInHospital_Y210': 0.23731146168158038,
 'DaysInHospital_Y211': 0.1710723543155184,
 'DaysInHospital_Y212': 0.236856544714515,
 'DaysInHospital_Y213': 0.16892919181367275,
 'DaysInHospital_Y214': 0.23558385967427445,
 'DaysInHospital_Y215': 0.67983989455717353,
 'DaysInHospital_Y22': 0.050406664146141519,
 'DaysInHospital_Y23': 0.079268960824596391,
 'DaysInHospital_Y24': 0.17473576356507209,
 'DaysInHospital_Y25': 0.13826840457010925,
 'DaysInHospital_Y26': 0.20653378434839736,
 'DaysInHospital_Y27': 0.16362134058527042,
 'DaysInHospital_Y28': 0.23918670396271141,
 'DaysInHospital_Y29': 0.17948641033261789,
 'LengthOfStay1-_2_weeks': 0.016002714471196423,
 'LengthOfStay12-26_weeks': -0.0005386616908135206,
 'LengthOfStay1_day': 0.017734250102598718,
 'LengthOfStay2-_4_weeks': 0.011378504058779076,
 'LengthOfStay26+_weeks': 0.0018575051880313092,
 'LengthOfStay2_days': 0.010544670413215319,
 'LengthOfStay3_days': 0.003834178163569857,
 'LengthOfStay4-_8_weeks': 0.015374103564451316,
 'LengthOfStay4_days': 0.0029952229839018997,
 'LengthOfStay5_days': 0.005178171758455947,
 'LengthOfStay6_days': 0.00097592601611427572,
 'LengthOfStay8-12_weeks': 0.0037639000845775205,
 #'LengthOfStaynan': nan,
 'PrimaryConditionGroupAMI': 0.030610758769220529,
 'PrimaryConditionGroupAPPCHOL': 0.0012128879780080107,
 'PrimaryConditionGroupARTHSPIN': -0.0035520633979465302,
 'PrimaryConditionGroupCANCRA': 0.0035063605670472044,
 'PrimaryConditionGroupCANCRB': 0.01046496422568827,
 'PrimaryConditionGroupCANCRM': 0.0039967413725477026,
 'PrimaryConditionGroupCATAST': 0.0085184472783841995,
 'PrimaryConditionGroupCHF': 0.030577436050487891,
 'PrimaryConditionGroupCOPD': 0.016411282973425607,
 'PrimaryConditionGroupFLaELEC': 0.0073563987878677017,
 'PrimaryConditionGroupFXDISLC': -0.010069047003729216,
 'PrimaryConditionGroupGIBLEED': 0.018035041532077942,
 'PrimaryConditionGroupGIOBSENT': -0.0014232966616945406,
 'PrimaryConditionGroupGYNEC1': -0.014752923195988022,
 'PrimaryConditionGroupGYNECA': -0.0050279477676819723,
 'PrimaryConditionGroupHEART2': 0.032848460309481527,
 'PrimaryConditionGroupHEART4': 0.011181089744136057,
 'PrimaryConditionGroupHEMTOL': 0.019245959221376487,
 'PrimaryConditionGroupHIPFX': 0.010136020107810244,
 'PrimaryConditionGroupINFEC4': -0.014684059391852832,
 'PrimaryConditionGroupLIVERDZ': -0.00084052826229429262,
 'PrimaryConditionGroupMETAB1': 0.0025547248267555832,
 'PrimaryConditionGroupMETAB3': -0.0057154033867015596,
 'PrimaryConditionGroupMISCHRT': 0.0032039733347676439,
 'PrimaryConditionGroupMISCL1': 0.0008906653902656513,
 'PrimaryConditionGroupMISCL5': 0.00024306204742741011,
 'PrimaryConditionGroupMSC2a3': -0.032360877340658872,
 'PrimaryConditionGroupNEUMENT': -0.0015903552139161674,
 'PrimaryConditionGroupODaBNCA': -0.012518443796833612,
 'PrimaryConditionGroupPERINTL': -0.0015112797623188408,
 'PrimaryConditionGroupPERVALV': 0.0022840476904212627,
 'PrimaryConditionGroupPNCRDZ': 0.00050839345982758529,
 'PrimaryConditionGroupPNEUM': 0.0068629547626261724,
 'PrimaryConditionGroupPRGNCY': 0.0059664273791333955,
 'PrimaryConditionGroupRENAL1': 0.0050715918738550936,
 'PrimaryConditionGroupRENAL2': 0.034338474947324282,
 'PrimaryConditionGroupRENAL3': 0.0025755303445003474,
 'PrimaryConditionGroupRESPR4': -0.017525071823173366,
 'PrimaryConditionGroupROAMI': 0.025019415660643404,
 'PrimaryConditionGroupSEIZURE': 0.011789752443471919,
 'PrimaryConditionGroupSEPSIS': 0.00399247371507927,
 'PrimaryConditionGroupSKNAUT': -0.007385328441171464,
 'PrimaryConditionGroupSTROKE': 0.011273321660608532,
 'PrimaryConditionGroupTRAUMA': -0.013109609509598077,
 'PrimaryConditionGroupUTI': -0.0013519147971185026,
 'dsfs0-_1_month': -0.057138676661269094,
 'dsfs1-_2_months': -0.0010176712515487575,
 'dsfs10-11_months': 0.022174259224654772,
 'dsfs11-12_months': 0.021351925291324389,
 'dsfs2-_3_months': 0.0034856194173231679,
 'dsfs3-_4_months': 0.0068630200084156025,
 'dsfs4-_5_months': 0.0092195808845035414,
 'dsfs5-_6_months': 0.0088008967415616896,
 'dsfs6-_7_months': 0.009868797199045461,
 'dsfs7-_8_months': 0.0089013577662562116,
 'dsfs8-_9_months': 0.010949392053089547,
 'dsfs9-10_months': 0.015966734567246895,
 'paydelay0': 0.0010521142646558442,
 'paydelay1': -0.00092969779605539191,
 'paydelay10': -0.0025502610280857766,
 'paydelay100': -0.0021948827214483672,
 'paydelay101': -0.0022543582507939283,
 'paydelay102': -0.00076754526042429954,
 'paydelay103': -0.000271468074138099,
 'paydelay104': 0.0014270951644715379,
 'paydelay105': 0.00026535315176085108,
 'paydelay106': -0.00030860347235817781,
 'paydelay107': 0.00057344716599231282,
 'paydelay108': -0.00043698867854749383,
 'paydelay109': 0.00068037487483748205,
 'paydelay11': 0.00037663254145464274,
 'paydelay110': -0.0010481655110123355,
 'paydelay111': -0.0004457543736642376,
 'paydelay112': -0.0023746402996821588,
 'paydelay113': -0.00090301425209026679,
 'paydelay114': -0.00076817015232317446,
 'paydelay115': -0.0010761821218718019,
 'paydelay116': 0.0017048715221576682,
 'paydelay117': 0.0038798220390708852,
 'paydelay118': 0.0017741585359746747,
 'paydelay119': 0.00028431984604136999,
 'paydelay12': -0.0020368549866071461,
 'paydelay120': -0.0036680198223440121,
 'paydelay121': -0.00065209103856827687,
 'paydelay122': 0.0012017206522238117,
 'paydelay123': 2.5550710863681468e-05,
 'paydelay124': 0.0008210810783944515,
 'paydelay125': 0.0032403904494617826,
 'paydelay126': 7.4339891084732973e-05,
 'paydelay127': 0.0024288760160666866,
 'paydelay128': 0.0020189089646596215,
 'paydelay129': -0.0013200063738423974,
 'paydelay13': -0.001636357434912249,
 'paydelay130': -0.000842517470032419,
 'paydelay131': 0.0020888212601223913,
 'paydelay132': 0.00099817479090601874,
 'paydelay133': 0.0014182099099480205,
 'paydelay134': -0.0011503125677089919,
 'paydelay135': -0.00083604595560671203,
 'paydelay136': -0.0023341381163196392,
 'paydelay137': 0.0014618489569780187,
 'paydelay138': 0.00050695046093760378,
 'paydelay139': -0.00076701557983496619,
 'paydelay14': -0.001893447121653207,
 'paydelay140': -0.00055311865636457642,
 'paydelay141': 0.0004516991751084062,
 'paydelay142': 0.00074921851263036259,
 'paydelay143': 0.0016783255939981708,
 'paydelay144': 0.00027601465640864249,
 'paydelay145': -0.0018700317583144521,
 'paydelay146': 0.00093681953931392628,
 'paydelay147': 0.0018947403226469138,
 'paydelay148': -0.0015966683779331588,
 'paydelay149': 0.0013937722871974827,
 'paydelay15': -0.0026777360291627781,
 'paydelay150': -0.00018838730092220093,
 'paydelay151': -0.0018954405801069163,
 'paydelay152': -0.00040537224202083261,
 'paydelay153': 0.00027323334292028072,
 'paydelay154': -0.00060902775054555425,
 'paydelay155': 1.5935013123199077e-05,
 'paydelay156': -0.0023379371073389975,
 'paydelay157': 0.0012608020158976373,
 'paydelay158': 0.00062167705798619403,
 'paydelay159': 0.0016857095281476435,
 'paydelay16': -0.0033853331634081251,
 'paydelay160': -0.0012190988503349804,
 'paydelay161': -0.0011328509624568633,
 'paydelay17': -0.0042140380542248076,
 'paydelay18': -0.0058557031557985228,
 'paydelay19': -0.0030912553793221312,
 'paydelay2': 0.0014038357567363574,
 'paydelay20': -0.0023940881501139616,
 'paydelay21': -0.0051321873247753662,
 'paydelay22': 0.00049039358967818865,
 'paydelay23': 0.00086120951685571886,
 'paydelay24': 0.0016050619844855205,
 'paydelay25': -0.003331019095847407,
 'paydelay26': 0.00085687351416086565,
 'paydelay27': 0.00023953372171437996,
 'paydelay28': 0.00018449120684821204,
 'paydelay29': 0.001819012213016514,
 'paydelay3': -0.00071354694818117237,
 'paydelay30': -0.00014423920271112382,
 'paydelay31': 0.0016133046090605315,
 'paydelay32': -0.00022755699537699975,
 'paydelay33': 0.0019784784514850682,
 'paydelay34': 0.00040667101071007508,
 'paydelay35': -0.0003943430495734369,
 'paydelay36': 0.0020434853382608996,
 'paydelay37': 0.0028807577930185951,
 'paydelay38': -0.00029577873301640806,
 'paydelay39': -0.00099496510649477003,
 'paydelay4': -0.0016567159620019646,
 'paydelay40': 0.0046244091154052855,
 'paydelay41': 0.0017429132865842193,
 'paydelay42': -0.0014398584785627537,
 'paydelay43': 0.00031796050710647188,
 'paydelay44': 0.00069441307348550919,
 'paydelay45': 0.0015827338196245828,
 'paydelay46': 0.0011842733231384802,
 'paydelay47': 0.0094826817560977449,
 'paydelay48': 0.0038598870396361995,
 'paydelay49': 0.0025892810697994884,
 'paydelay5': -0.00095716436296253355,
 'paydelay50': 0.0013784816139350302,
 'paydelay51': 0.0052888330038332373,
 'paydelay52': 0.0043719061139536365,
 'paydelay53': 0.00076936561131082882,
 'paydelay54': 0.0030249319260404715,
 'paydelay55': 0.0037535297572629913,
 'paydelay56': -0.00037001726288951456,
 'paydelay57': 0.00094027381845838032,
 'paydelay58': 0.00091308544268382948,
 'paydelay59': -0.0023924908068606713,
 'paydelay6': -0.001077564540917901,
 'paydelay60': -0.0019681620255400196,
 'paydelay61': 0.0031032076237671475,
 'paydelay62': -7.0868717864139321e-05,
 'paydelay63': -0.003745593432843248,
 'paydelay64': 6.3218315869063629e-05,
 'paydelay65': 0.001953719689020343,
 'paydelay66': -0.0028456409906599104,
 'paydelay67': -0.0035566623552632736,
 'paydelay68': 0.0013236523155071083,
 'paydelay69': 0.00022040552408937974,
 'paydelay7': -0.00032590406651093306,
 'paydelay70': -0.0038638803734610209,
 'paydelay71': -0.0033764190361944132,
 'paydelay72': -0.0038707936862452546,
 'paydelay73': -0.0032796771173344481,
 'paydelay74': -0.00091994208203768587,
 'paydelay75': -0.00010530661341791988,
 'paydelay76': 3.6871601203532467e-05,
 'paydelay77': -0.0012384775983214993,
 'paydelay78': -0.0011052492004519157,
 'paydelay79': -0.004198329894719203,
 'paydelay8': -0.0019500907711411293,
 'paydelay80': -0.0012343309007495648,
 'paydelay81': -0.0017991867375654399,
 'paydelay82': -0.0010852277906497831,
 'paydelay83': 0.00014023613100656252,
 'paydelay84': -0.0030434918637921781,
 'paydelay85': -0.00065015308895255647,
 'paydelay86': -0.0025513891738748111,
 'paydelay87': 0.00065330255159183777,
 'paydelay88': -0.001710048085560648,
 'paydelay89': 0.0011907776389895187,
 'paydelay9': -0.00041272688407094002,
 'paydelay90': -0.00066719121993306972,
 'paydelay91': -0.00048590370692681736,
 'paydelay92': -4.2848067783431625e-05,
 'paydelay93': -0.00099883873192105062,
 'paydelay94': -0.0013828011249261995,
 'paydelay95': -0.00083475409542182663,
 'paydelay96': -2.2257659964384273e-05,
 'paydelay97': 0.00078055295242069537,
 'paydelay98': 0.0009529945677058465,
 'paydelay99': 0.0009572416524036379,
 #'paydelaynan': nan,
 'placesvcAmbulance': -0.0088951291144559845,
 'placesvcHome': 0.022999478492210988,
 'placesvcIndependent_Lab': -0.0091750319900114063,
 'placesvcInpatient_Hospital': 0.067024686882097659,
 'placesvcOffice': -0.041390493806630409,
 'placesvcOther': 0.032556303510249691,
 'placesvcOutpatient_Hospital': 0.0073223258311093725,
 'placesvcUrgent_Care': 0.02925019299131448,
 'sexF': 0.0058870678746491449,
 'sexM': -0.0058870678746491207,
 'specialtyAnesthesiology': 0.00067542493953508763,
 'specialtyDiagnostic_Imaging': 0.013012588534897708,
 'specialtyEmergency': 0.018652276647575378,
 'specialtyGeneral_Practice': -0.03317219944350918,
 'specialtyInternal': 0.031778661136686516,
 'specialtyLaboratory': -0.0082867311504348142,
 'specialtyObstetrics_and_Gynecology': -0.01497379196138463,
 'specialtyOther': 0.014017596989319909,
 'specialtyPathology': -0.0028972566559410341,
 'specialtyPediatrics': -0.041589311514704998,
 'specialtyRehabilitation': -0.0042221748355297222,
 'specialtySurgery': 0.0047687295186776306}

===================sort ==============================
[('DaysInHospital_Y20', -0.6794634231536308),
 ('CharlsonIndex0', -0.13824233051485832),
 ('AgeAtFirstClaim40-49', -0.07082970657691501),
 ('dsfs0-_1_month', -0.057138676661269094),
 ('AgeAtFirstClaim10-19', -0.05312058521422108),
 ('AgeAtFirstClaim0-9', -0.051990535358995144),
 ('AgeAtFirstClaim50-59', -0.051728956041176576),
 ('specialtyPediatrics', -0.041589311514705),
 ('placesvcOffice', -0.04139049380663041),
 ('AgeAtFirstClaim30-39', -0.03412172330118397),
 ('specialtyGeneral_Practice', -0.03317219944350918),
 ('PrimaryConditionGroupMSC2a3', -0.03236087734065887),
 ('AgeAtFirstClaim20-29', -0.020055519475893367),
 ('PrimaryConditionGroupRESPR4', -0.017525071823173366),
 ('specialtyObstetrics_and_Gynecology', -0.01497379196138463),
 ('PrimaryConditionGroupGYNEC1', -0.014752923195988022),
 ('PrimaryConditionGroupINFEC4', -0.014684059391852832),
 ('PrimaryConditionGroupTRAUMA', -0.013109609509598077),
 ('AgeAtFirstClaim60-69', -0.012948134422449459),
 ('PrimaryConditionGroupODaBNCA', -0.012518443796833612),
 ('PrimaryConditionGroupFXDISLC', -0.010069047003729216),
 ('placesvcIndependent_Lab', -0.009175031990011406),
 ('placesvcAmbulance', -0.008895129114455985),
 ('DaysInHospital_Y21', -0.00873393883616769),
 ('specialtyLaboratory', -0.008286731150434814),
 ('PrimaryConditionGroupSKNAUT', -0.007385328441171464),
 ('sexM', -0.005887067874649121),
 ('paydelay18', -0.005855703155798523),
 ('PrimaryConditionGroupMETAB3', -0.00571540338670156),
 ('paydelay21', -0.005132187324775366),
 ('PrimaryConditionGroupGYNECA', -0.005027947767681972),
 ('specialtyRehabilitation', -0.004222174835529722),
 ('paydelay17', -0.004214038054224808),
 ('paydelay79', -0.004198329894719203),
 ('paydelay72', -0.0038707936862452546),
 ('paydelay70', -0.003863880373461021),
 ('paydelay63', -0.003745593432843248),
 ('paydelay120', -0.003668019822344012),
 ('paydelay67', -0.0035566623552632736),
 ('PrimaryConditionGroupARTHSPIN', -0.00355206339794653),
 ('paydelay16', -0.003385333163408125),
 ('paydelay71', -0.003376419036194413),
 ('paydelay25', -0.003331019095847407),
 ('paydelay73', -0.003279677117334448),
 ('paydelay19', -0.003091255379322131),
 ('paydelay84', -0.003043491863792178),
 ('specialtyPathology', -0.002897256655941034),
 ('paydelay66', -0.0028456409906599104),
 ('paydelay15', -0.002677736029162778),
 ('paydelay86', -0.002551389173874811),
 ('paydelay10', -0.0025502610280857766),
 ('paydelay20', -0.0023940881501139616),
 ('paydelay59', -0.0023924908068606713),
 ('paydelay112', -0.0023746402996821588),
 ('paydelay156', -0.0023379371073389975),
 ('paydelay136', -0.002334138116319639),
 ('paydelay101', -0.0022543582507939283),
 ('paydelay100', -0.002194882721448367),
 ('paydelay12', -0.002036854986607146),
 ('paydelay60', -0.0019681620255400196),
 ('paydelay8', -0.0019500907711411293),
 ('paydelay151', -0.0018954405801069163),
 ('paydelay14', -0.001893447121653207),
 ('paydelay145', -0.0018700317583144521),
 ('paydelay81', -0.0017991867375654399),
 ('paydelay88', -0.001710048085560648),
 ('paydelay4', -0.0016567159620019646),
 ('paydelay13', -0.001636357434912249),
 ('paydelay148', -0.0015966683779331588),
 ('PrimaryConditionGroupNEUMENT', -0.0015903552139161674),
 ('PrimaryConditionGroupPERINTL', -0.0015112797623188408),
 ('paydelay42', -0.0014398584785627537),
 ('PrimaryConditionGroupGIOBSENT', -0.0014232966616945406),
 ('paydelay94', -0.0013828011249261995),
 ('PrimaryConditionGroupUTI', -0.0013519147971185026),
 ('paydelay129', -0.0013200063738423974),
 ('paydelay77', -0.0012384775983214993),
 ('paydelay80', -0.0012343309007495648),
 ('paydelay160', -0.0012190988503349804),
 ('paydelay134', -0.001150312567708992),
 ('paydelay161', -0.0011328509624568633),
 ('paydelay78', -0.0011052492004519157),
 ('paydelay82', -0.001085227790649783),
 ('paydelay6', -0.001077564540917901),
 ('paydelay115', -0.001076182121871802),
 ('paydelay110', -0.0010481655110123355),
 ('dsfs1-_2_months', -0.0010176712515487575),
 ('paydelay93', -0.0009988387319210506),
 ('paydelay39', -0.00099496510649477),
 ('paydelay5', -0.0009571643629625335),
 ('paydelay1', -0.0009296977960553919),
 ('paydelay74', -0.0009199420820376859),
 ('paydelay113', -0.0009030142520902668),
 ('paydelay130', -0.000842517470032419),
 ('PrimaryConditionGroupLIVERDZ', -0.0008405282622942926),
 ('paydelay135', -0.000836045955606712),
 ('paydelay95', -0.0008347540954218266),
 ('paydelay114', -0.0007681701523231745),
 ('paydelay102', -0.0007675452604242995),
 ('paydelay139', -0.0007670155798349662),
 ('paydelay3', -0.0007135469481811724),
 ('paydelay90', -0.0006671912199330697),
 ('paydelay121', -0.0006520910385682769),
 ('paydelay85', -0.0006501530889525565),
 ('paydelay154', -0.0006090277505455542),
 ('paydelay140', -0.0005531186563645764),
 ('LengthOfStay12-26_weeks', -0.0005386616908135206),
 ('paydelay91', -0.00048590370692681736),
 ('paydelay111', -0.0004457543736642376),
 ('paydelay108', -0.0004369886785474938),
 ('paydelay9', -0.00041272688407094),
 ('paydelay152', -0.0004053722420208326),
 ('paydelay35', -0.0003943430495734369),
 ('paydelay56', -0.00037001726288951456),
 ('paydelay7', -0.00032590406651093306),
 ('paydelay106', -0.0003086034723581778),
 ('paydelay38', -0.00029577873301640806),
 ('paydelay103', -0.000271468074138099),
 ('paydelay32', -0.00022755699537699975),
 ('paydelay150', -0.00018838730092220093),
 ('paydelay30', -0.00014423920271112382),
 ('paydelay75', -0.00010530661341791988),
 ('paydelay62', -7.086871786413932e-05),
 ('paydelay92', -4.2848067783431625e-05),
 ('paydelay96', -2.2257659964384273e-05),
 ('paydelay155', 1.5935013123199077e-05),
 ('paydelay123', 2.5550710863681468e-05),
 ('paydelay76', 3.687160120353247e-05),
 ('paydelay64', 6.321831586906363e-05),
 ('paydelay126', 7.433989108473297e-05),
 ('paydelay83', 0.00014023613100656252),
 ('paydelay28', 0.00018449120684821204),
 ('paydelay69', 0.00022040552408937974),
 ('paydelay27', 0.00023953372171437996),
 ('PrimaryConditionGroupMISCL5', 0.00024306204742741011),
 ('paydelay105', 0.0002653531517608511),
 ('paydelay153', 0.0002732333429202807),
 ('paydelay144', 0.0002760146564086425),
 ('paydelay119', 0.00028431984604137),
 ('paydelay43', 0.0003179605071064719),
 ('paydelay11', 0.00037663254145464274),
 ('paydelay34', 0.0004066710107100751),
 ('paydelay141', 0.0004516991751084062),
 ('paydelay22', 0.0004903935896781886),
 ('paydelay138', 0.0005069504609376038),
 ('PrimaryConditionGroupPNCRDZ', 0.0005083934598275853),
 ('paydelay107', 0.0005734471659923128),
 ('paydelay158', 0.000621677057986194),
 ('paydelay87', 0.0006533025515918378),
 ('specialtyAnesthesiology', 0.0006754249395350876),
 ('paydelay109', 0.000680374874837482),
 ('paydelay44', 0.0006944130734855092),
 ('paydelay142', 0.0007492185126303626),
 ('paydelay53', 0.0007693656113108288),
 ('paydelay97', 0.0007805529524206954),
 ('paydelay124', 0.0008210810783944515),
 ('paydelay26', 0.0008568735141608657),
 ('paydelay23', 0.0008612095168557189),
 ('PrimaryConditionGroupMISCL1', 0.0008906653902656513),
 ('paydelay58', 0.0009130854426838295),
 ('paydelay146', 0.0009368195393139263),
 ('paydelay57', 0.0009402738184583803),
 ('paydelay98', 0.0009529945677058465),
 ('paydelay99', 0.0009572416524036379),
 ('LengthOfStay6_days', 0.0009759260161142757),
 ('paydelay132', 0.0009981747909060187),
 ('paydelay0', 0.0010521142646558442),
 ('paydelay46', 0.0011842733231384802),
 ('paydelay89', 0.0011907776389895187),
 ('paydelay122', 0.0012017206522238117),
 ('PrimaryConditionGroupAPPCHOL', 0.0012128879780080107),
 ('paydelay157', 0.0012608020158976373),
 ('paydelay68', 0.0013236523155071083),
 ('paydelay50', 0.0013784816139350302),
 ('paydelay149', 0.0013937722871974827),
 ('paydelay2', 0.0014038357567363574),
 ('paydelay133', 0.0014182099099480205),
 ('paydelay104', 0.001427095164471538),
 ('paydelay137', 0.0014618489569780187),
 ('paydelay45', 0.0015827338196245828),
 ('paydelay24', 0.0016050619844855205),
 ('paydelay31', 0.0016133046090605315),
 ('paydelay143', 0.0016783255939981708),
 ('paydelay159', 0.0016857095281476435),
 ('paydelay116', 0.0017048715221576682),
 ('paydelay41', 0.0017429132865842193),
 ('paydelay118', 0.0017741585359746747),
 ('paydelay29', 0.001819012213016514),
 ('LengthOfStay26+_weeks', 0.0018575051880313092),
 ('paydelay147', 0.0018947403226469138),
 ('paydelay65', 0.001953719689020343),
 ('paydelay33', 0.001978478451485068),
 ('paydelay128', 0.0020189089646596215),
 ('paydelay36', 0.0020434853382608996),
 ('paydelay131', 0.0020888212601223913),
 ('PrimaryConditionGroupPERVALV', 0.0022840476904212627),
 ('paydelay127', 0.0024288760160666866),
 ('PrimaryConditionGroupMETAB1', 0.0025547248267555832),
 ('PrimaryConditionGroupRENAL3', 0.0025755303445003474),
 ('paydelay49', 0.0025892810697994884),
 ('paydelay37', 0.002880757793018595),
 ('LengthOfStay4_days', 0.0029952229839018997),
 ('paydelay54', 0.0030249319260404715),
 ('paydelay61', 0.0031032076237671475),
 ('PrimaryConditionGroupMISCHRT', 0.003203973334767644),
 ('paydelay125', 0.0032403904494617826),
 ('dsfs2-_3_months', 0.003485619417323168),
 ('PrimaryConditionGroupCANCRA', 0.0035063605670472044),
 ('paydelay55', 0.0037535297572629913),
 ('LengthOfStay8-12_weeks', 0.0037639000845775205),
 ('LengthOfStay3_days', 0.003834178163569857),
 ('paydelay48', 0.0038598870396361995),
 ('paydelay117', 0.003879822039070885),
 ('PrimaryConditionGroupSEPSIS', 0.00399247371507927),
 ('PrimaryConditionGroupCANCRM', 0.003996741372547703),
 ('paydelay52', 0.0043719061139536365),
 ('paydelay40', 0.0046244091154052855),
 ('specialtySurgery', 0.0047687295186776306),
 ('PrimaryConditionGroupRENAL1', 0.005071591873855094),
 ('LengthOfStay5_days', 0.005178171758455947),
 ('paydelay51', 0.005288833003833237),
 ('sexF', 0.005887067874649145),
 ('PrimaryConditionGroupPRGNCY', 0.0059664273791333955),
 ('PrimaryConditionGroupPNEUM', 0.006862954762626172),
 ('dsfs3-_4_months', 0.0068630200084156025),
 ('placesvcOutpatient_Hospital', 0.0073223258311093725),
 ('PrimaryConditionGroupFLaELEC', 0.007356398787867702),
 ('PrimaryConditionGroupCATAST', 0.0085184472783842),
 ('dsfs5-_6_months', 0.00880089674156169),
 ('dsfs7-_8_months', 0.008901357766256212),
 ('dsfs4-_5_months', 0.009219580884503541),
 ('paydelay47', 0.009482681756097745),
 ('dsfs6-_7_months', 0.009868797199045461),
 ('PrimaryConditionGroupHIPFX', 0.010136020107810244),
 ('PrimaryConditionGroupCANCRB', 0.01046496422568827),
 ('LengthOfStay2_days', 0.010544670413215319),
 ('CharlsonIndex5+', 0.010936434562424569),
 ('dsfs8-_9_months', 0.010949392053089547),
 ('PrimaryConditionGroupHEART4', 0.011181089744136057),
 ('PrimaryConditionGroupSTROKE', 0.011273321660608532),
 ('LengthOfStay2-_4_weeks', 0.011378504058779076),
 ('PrimaryConditionGroupSEIZURE', 0.01178975244347192),
 ('specialtyDiagnostic_Imaging', 0.013012588534897708),
 ('specialtyOther', 0.01401759698931991),
 ('LengthOfStay4-_8_weeks', 0.015374103564451316),
 ('dsfs9-10_months', 0.015966734567246895),
 ('LengthOfStay1-_2_weeks', 0.016002714471196423),
 ('PrimaryConditionGroupCOPD', 0.016411282973425607),
 ('LengthOfStay1_day', 0.017734250102598718),
 ('PrimaryConditionGroupGIBLEED', 0.018035041532077942),
 ('specialtyEmergency', 0.018652276647575378),
 ('PrimaryConditionGroupHEMTOL', 0.019245959221376487),
 ('dsfs11-12_months', 0.02135192529132439),
 ('dsfs10-11_months', 0.022174259224654772),
 ('placesvcHome', 0.022999478492210988),
 ('PrimaryConditionGroupROAMI', 0.025019415660643404),
 ('placesvcUrgent_Care', 0.02925019299131448),
 ('PrimaryConditionGroupCHF', 0.03057743605048789),
 ('PrimaryConditionGroupAMI', 0.03061075876922053),
 ('specialtyInternal', 0.031778661136686516),
 ('placesvcOther', 0.03255630351024969),
 ('PrimaryConditionGroupHEART2', 0.03284846030948153),
 ('PrimaryConditionGroupRENAL2', 0.03433847494732428),
 ('CharlsonIndex3-4', 0.0472330686553357),
 ('DaysInHospital_Y22', 0.05040666414614152),
 ('AgeAtFirstClaim70-79', 0.06431186328261086),
 ('placesvcInpatient_Hospital', 0.06702468688209766),
 ('DaysInHospital_Y23', 0.07926896082459639),
 ('CharlsonIndex1-2', 0.12582855508262986),
 ('AgeAtFirstClaim80+', 0.13442560590738883),
 ('DaysInHospital_Y25', 0.13826840457010925),
 ('DaysInHospital_Y27', 0.16362134058527042),
 ('DaysInHospital_Y213', 0.16892919181367275),
 ('DaysInHospital_Y211', 0.1710723543155184),
 ('DaysInHospital_Y24', 0.1747357635650721),
 ('DaysInHospital_Y29', 0.1794864103326179),
 ('DaysInHospital_Y26', 0.20653378434839736),
 ('DaysInHospital_Y214', 0.23558385967427445),
 ('DaysInHospital_Y212', 0.236856544714515),
 ('DaysInHospital_Y210', 0.23731146168158038),
 ('DaysInHospital_Y28', 0.2391867039627114),
 ('DaysInHospital_Y215', 0.6798398945571735)]



"""
def TestDays2():
    """
    ['AgeAtFirstClaim80+'] => 0.76758899605
    ['CharlsonIndex1-2'] => 0.76758899605
    ['CharlsonIndex0'] => 0.76758899605
    ['AgeAtFirstClaim80+', 'CharlsonIndex1-2'] => 0.76758899605
    ['AgeAtFirstClaim80+', 'CharlsonIndex0'] => 0.76758899605
    ['CharlsonIndex1-2', 'CharlsonIndex0'] => 0.76758899605
    ['AgeAtFirstClaim80+', 'CharlsonIndex1-2', 'CharlsonIndex0'] => 0.76758899605    
    """
    from sklearn import cross_validation
    from sklearn import svm
    import itertools
    from sklearn.ensemble import RandomForestClassifier
    #itertools.permutations(["DaysInHospital_Y20","DaysInHospital_Y215","DaysInHospital_Y212","DaysInHospital_Y26","DaysInHospital_Y28","DaysInHospital_Y210", "DaysInHospital_Y214"])
    m=pd.read_pickle("featuresSet")
    features= list(m.columns)
    """
    features=[
     "AgeAtFirstClaim80+",
     "CharlsonIndex1-2",
     "CharlsonIndex0",
    ]
    """
    
    combs = []
    for i in xrange(10,12): #xrange(100, len(features)+1):
        #combs.append(i)
        els = [list(x) for x in itertools.combinations(features, i)]
        combs.extend(els)
    #return combs
    print "features finsihed"
    result={}    
    for i in combs:
        X=m[i]
        Y=m["DaysInHospital_Y2"]
        X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,Y,test_size=0.95,random_state=0)
        #clf = RandomForestClassifier(n_estimators=100)
        #clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
        #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        clf = svm.LinearSVC()
        clf = clf.fit(X_train, y_train)
        score=clf.score(X_test, y_test)   
        print i,"=>",score
        result[str(i)]=score
    return result   
   
"""
#dict
{"['DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.86902890096020868,
 "['DaysInHospital_Y20']": 0.84933654001577219,
 "['DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87493122123333189,
 "['DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.85523886028889529,
 "['DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.79318367726798489,
 "['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88486312004976564,
 "['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86517075910532903,
 "['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80311557608441864,
 "['DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.78342321513998214,
 "['DaysInHospital_Y210']": 0.77349131632354839,
 "['DaysInHospital_Y212', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.873001333940709,
 "['DaysInHospital_Y212', 'DaysInHospital_Y20']": 0.85330897299627251,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87890365421383221,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.8592112932693956,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.79715611024848521,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88883555303026596,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86914319208582935,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80708800906491895,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.78739564812048246,
 "['DaysInHospital_Y212', 'DaysInHospital_Y210']": 0.77746374930404871,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87582922293463694,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20']": 0.85613686199020034,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88173154320776004,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.86203918226332343,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.79998399924241315,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89166344202419379,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.87197108107975729,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80991589805884689,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.79022353711441029,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210']": 0.78029163829797654,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y215']": 0.79408167896928994,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88576112175107069,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86606876080663409,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80401357778572369,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28']": 0.78432121684128719,
 "['DaysInHospital_Y212', 'DaysInHospital_Y214']": 0.77438931802485345,
 "['DaysInHospital_Y212', 'DaysInHospital_Y215']": 0.79125378997536211,
 "['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88293323275714275,
 "['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86324087181270626,
 "['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80118568879179586,
 "['DaysInHospital_Y212', 'DaysInHospital_Y28']": 0.78149332784735925,
 "['DaysInHospital_Y212']": 0.77156142903092551,
 "['DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87185678995413662,
 "['DaysInHospital_Y214', 'DaysInHospital_Y20']": 0.85216442900970002,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87775911022725972,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.85806674928282323,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.79601156626191283,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88769100904369347,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86799864809925698,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80594346507834658,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.78625110413390997,
 "['DaysInHospital_Y214', 'DaysInHospital_Y210']": 0.77631920531747622,
 "['DaysInHospital_Y214', 'DaysInHospital_Y215']": 0.79010924598878962,
 "['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88178868877057037,
 "['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.86209632782613377,
 "['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.80004114480522348,
 "['DaysInHospital_Y214', 'DaysInHospital_Y28']": 0.78034878386078688,
 "['DaysInHospital_Y214']": 0.77041688504435313,
 "['DaysInHospital_Y215']": 0.78728135699486179,
 "['DaysInHospital_Y26', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88362714316269664,
 "['DaysInHospital_Y26', 'DaysInHospital_Y20']": 0.86393478221826014,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88952946343581984,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.86983710249138324,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.80778191947047284,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89946136225225359,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.87976900130781699,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.81771381828690659,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.7980214573424701,
 "['DaysInHospital_Y26', 'DaysInHospital_Y210']": 0.78808955852603635,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88759957614319696,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y20']": 0.86790721519876046,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89350189641632016,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.87380953547188356,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.81175435245097316,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.90343379523275391,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.88374143428831731,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.82168625126740691,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.80199389032297042,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210']": 0.79206199150653667,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.8904274651371249,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20']": 0.87073510419268829,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89632978541024799,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.87663742446581139,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.8145822414449011,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.90626168422668174,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.88656932328224525,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.82451414026133485,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.80482177931689824,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210']": 0.7948898805004645,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y215']": 0.80867992117177789,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.90035936395355864,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.88066700300912204,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.81861181998821164,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28']": 0.79891945904377515,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214']": 0.7889875602273414,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y215']": 0.80585203217785006,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.8975314749596307,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.87783911401519421,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.81578393099428381,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28']": 0.79609157004984721,
 "['DaysInHospital_Y26', 'DaysInHospital_Y212']": 0.78615967123341346,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.88645503215662458,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y20']": 0.86676267121218797,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89235735242974767,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']": 0.87266499148531118,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']": 0.81060980846440078,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.90228925124618142,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.88259689030174493,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.82054170728083453,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']": 0.80084934633639793,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210']": 0.79091744751996418,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y215']": 0.80470748819127758,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89638693097305833,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.87669457002862172,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.81463938700771144,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28']": 0.79494702606327483,
 "['DaysInHospital_Y26', 'DaysInHospital_Y214']": 0.78501512724684108,
 "['DaysInHospital_Y26', 'DaysInHospital_Y215']": 0.80187959919734975,
 "['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.89355904197913039,
 "['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.87386668103469389,
 "['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.81181149801378349,
 "['DaysInHospital_Y26', 'DaysInHospital_Y28']": 0.792119137069347,
 "['DaysInHospital_Y26']": 0.78218723825291325,
 "['DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']": 0.87896079977664243,
 "['DaysInHospital_Y28', 'DaysInHospital_Y20']": 0.85926843883220594,
 "['DaysInHospital_Y28', 'DaysInHospital_Y215']": 0.79721325581129554,
 "['DaysInHospital_Y28']": 0.77752089486685905}
sort dict
[("['DaysInHospital_Y214']", 0.77041688504435313),
 ("['DaysInHospital_Y212']", 0.77156142903092551),
 ("['DaysInHospital_Y210']", 0.77349131632354839),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214']", 0.77438931802485345),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210']", 0.77631920531747622),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210']", 0.77746374930404871),
 ("['DaysInHospital_Y28']", 0.77752089486685905),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210']",
  0.78029163829797654),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y28']", 0.78034878386078688),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y28']", 0.78149332784735925),
 ("['DaysInHospital_Y26']", 0.78218723825291325),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y28']", 0.78342321513998214),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28']",
  0.78432121684128719),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214']", 0.78501512724684108),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212']", 0.78615967123341346),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.78625110413390997),
 ("['DaysInHospital_Y215']", 0.78728135699486179),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.78739564812048246),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210']", 0.78808955852603635),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214']",
  0.7889875602273414),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y215']", 0.79010924598878962),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.79022353711441029),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210']",
  0.79091744751996418),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y215']", 0.79125378997536211),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210']",
  0.79206199150653667),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y28']", 0.792119137069347),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y215']", 0.79318367726798489),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y215']",
  0.79408167896928994),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210']",
  0.7948898805004645),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28']",
  0.79494702606327483),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.79601156626191283),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28']",
  0.79609157004984721),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.79715611024848521),
 ("['DaysInHospital_Y28', 'DaysInHospital_Y215']", 0.79721325581129554),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.7980214573424701),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28']",
  0.79891945904377515),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.79998399924241315),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80004114480522348),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.80084934633639793),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80118568879179586),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y215']", 0.80187959919734975),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.80199389032297042),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80311557608441864),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80401357778572369),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y215']",
  0.80470748819127758),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28']",
  0.80482177931689824),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y215']",
  0.80585203217785006),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80594346507834658),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80708800906491895),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.80778191947047284),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y215']",
  0.80867992117177789),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.80991589805884689),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.81060980846440078),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.81175435245097316),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.81181149801378349),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y215']",
  0.8145822414449011),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.81463938700771144),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.81578393099428381),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.81771381828690659),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.81861181998821164),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.82054170728083453),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.82168625126740691),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y215']",
  0.82451414026133485),
 ("['DaysInHospital_Y20']", 0.84933654001577219),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y20']", 0.85216442900970002),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y20']", 0.85330897299627251),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y20']", 0.85523886028889529),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20']",
  0.85613686199020034),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.85806674928282323),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.8592112932693956),
 ("['DaysInHospital_Y28', 'DaysInHospital_Y20']", 0.85926843883220594),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.86203918226332343),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86209632782613377),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86324087181270626),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y20']", 0.86393478221826014),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86517075910532903),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86606876080663409),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y20']",
  0.86676267121218797),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y20']",
  0.86790721519876046),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86799864809925698),
 ("['DaysInHospital_Y20', 'DaysInHospital_Y215']", 0.86902890096020868),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.86914319208582935),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.86983710249138324),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20']",
  0.87073510419268829),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87185678995413662),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.87197108107975729),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.87266499148531118),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.873001333940709),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.87380953547188356),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.87386668103469389),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87493122123333189),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87582922293463694),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20']",
  0.87663742446581139),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.87669457002862172),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87775911022725972),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.87783911401519421),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87890365421383221),
 ("['DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.87896079977664243),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.87976900130781699),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.88066700300912204),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88173154320776004),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88178868877057037),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.88259689030174493),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88293323275714275),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88362714316269664),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.88374143428831731),
 ("['DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88486312004976564),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88576112175107069),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88645503215662458),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20']",
  0.88656932328224525),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88759957614319696),
 ("['DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88769100904369347),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88883555303026596),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.88952946343581984),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.8904274651371249),
 ("['DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89166344202419379),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89235735242974767),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89350189641632016),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89355904197913039),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89632978541024799),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89638693097305833),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.8975314749596307),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.89946136225225359),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.90035936395355864),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.90228925124618142),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.90343379523275391),
 ("['DaysInHospital_Y26', 'DaysInHospital_Y212', 'DaysInHospital_Y214', 'DaysInHospital_Y210', 'DaysInHospital_Y28', 'DaysInHospital_Y20', 'DaysInHospital_Y215']",
  0.90626168422668174)]



"""


def madDataTest():
    mad=pd.read_csv("../mad_mark/Data_Set1.csv")
    mad.YEAR_t.replace("Y2",2,inplace=True)
    mad.YEAR_t.replace("Y1",1,inplace=True)
    mad.YEAR_t.replace("Y3",3,inplace=True)
    #mad.YEAR_t
    #mad.DaysInHospital.corr(mad.labCount_ave)
    for i in mad.columns:
        print i,"=",mad.DaysInHospital.corr(mad[i])
    """
    MemberID_t = -0.00430663479898
    YEAR_t = -0.00927449651427
    ClaimsTruncated = 0.164786506045
    DaysInHospital = 1.0
    trainset = nan
    age_05 = -0.056000826995
    age_15 = -0.0601445986863
    age_25 = -0.00794499300771
    age_35 = -0.0307292866905
    age_45 = -0.0562487273939
    age_55 = -0.0390754357467
    age_65 = 0.00590304481323
    age_75 = 0.0710671217519
    age_85 = 0.101658629313
    age_MISS = 0.0817565359938
    sexMALE = -0.0699930216563
    sexFEMALE = -0.0389861128079
    sexMISS = 0.135082061456
    no_Claims = 0.191122215283
    no_Providers = 0.178687347617
    no_Vendors = 0.173838642771
    no_PCPs = 0.0121198329051
    no_PlaceSvcs = 0.140370771264
    no_Specialities = 0.137277180517
    no_PrimaryConditionGroups = 0.175969203523
    no_ProcedureGroups = 0.128295617971
    PayDelay_max = 0.0992427451976
    PayDelay_min = -0.0630307029362
    PayDelay_ave = -0.00278160937172
    PayDelay_stdev = 0.0608644578676
    LOS_max = 0.0797714260777
    LOS_min = 0.0667063794365
    LOS_ave = 0.0752849145203
    LOS_stdev = 0.0554133074581
    LOS_TOT_UNKNOWN = 0.186738680016
    LOS_TOT_SUPRESSED = 0.0666110899297
    LOS_TOT_KNOWN = 0.0998976373244
    dsfs_max = 0.117510833593
    dsfs_min = 0.0264104171893
    dsfs_range = 0.116861895669
    dsfs_ave = 0.115028853457
    dsfs_stdev = 0.0790663596738
    CharlsonIndexI_max = 0.152195922287
    CharlsonIndexI_min = 0.0969224672947
    CharlsonIndexI_ave = 0.128451989723
    CharlsonIndexI_range = 0.113541948358
    CharlsonIndexI_stdev = 0.0933374209541
    pcg1 = 0.0739520510236
    pcg2 = 0.0701540027474
    pcg3 = 0.073081058508
    pcg4 = 0.0598992069914
    pcg5 = 0.0206281638522
    pcg6 = 0.0631887218619
    pcg7 = 0.0358078824202
    pcg8 = 0.0802693288378
    pcg9 = 0.0184279963971
    pcg10 = 0.014268701358
    pcg11 = 0.0541839153383
    pcg12 = 0.0433781039322
    pcg13 = 0.088400528648
    pcg14 = 0.0461191112097
    pcg15 = 0.0117985040715
    pcg16 = 0.0402131926482
    pcg17 = 0.0598554861521
    pcg18 = 0.0187439344017
    pcg19 = 0.042720851816
    pcg20 = 0.00379793739162
    pcg21 = 0.0748739537197
    pcg22 = 0.0392314138858
    pcg23 = 0.0630112715716
    pcg24 = 0.0433466926681
    pcg25 = 0.044333060569
    pcg26 = 0.016335777267
    pcg27 = 0.0697714174108
    pcg28 = 0.0141534115619
    pcg29 = 0.0456076464901
    pcg30 = 0.0288611920168
    pcg31 = 0.0673984521504
    pcg32 = 0.0223942847207
    pcg33 = 0.028283477293
    pcg34 = 0.00914701124468
    pcg35 = 0.029335526754
    pcg36 = 0.0140463023075
    pcg37 = 0.0224534022296
    pcg38 = 0.0186871861388
    pcg39 = 0.0169385080861
    pcg40 = 0.0109552477521
    pcg41 = 0.0187205316449
    pcg42 = 0.00491891392811
    pcg43 = 0.0017518099961
    pcg44 = 0.00818767366843
    pcg45 = 0.0140123301906
    pcg46 = 0.00883582358705
    sp1 = 0.152888243685
    sp2 = 0.122094967867
    sp3 = 0.0640035275466
    sp4 = 0.0737214001397
    sp5 = 0.143974908375
    sp6 = 0.110255300723
    sp7 = 0.0551049348415
    sp8 = -0.0238768672022
    sp9 = 0.0186497755297
    sp10 = 0.0217599186855
    sp11 = 0.0396079256916
    sp12 = 0.0272451851112
    sp13 = 0.0310359051973
    pg1 = 0.180603203581
    pg2 = 0.130500472696
    pg3 = 0.0989434232062
    pg4 = 0.104746767113
    pg5 = 0.130613637835
    pg6 = 0.0824205260144
    pg7 = 0.0558458721711
    pg8 = 0.0494179291948
    pg9 = 0.0338332976682
    pg10 = 0.0191526309919
    pg11 = 0.0243548858649
    pg12 = 0.0299536739902
    pg13 = 0.0353286480335
    pg14 = 0.00706503580015
    pg15 = 0.021954660059
    pg16 = 0.0267342776726
    pg17 = 0.00386111888478
    pg18 = 0.0107106095828
    ps1 = 0.141355718628
    ps2 = 0.122261227779
    ps3 = 0.099203413729
    ps4 = 0.0534632445368
    ps5 = 0.11766203886
    ps6 = 0.0131791248721
    ps7 = 0.0638188178692
    ps8 = 0.0367284858509
    ps9 = 0.0257697994186
    drugCount_max = 0.144863776508
    drugCount_min = 0.0840087974342
    drugCount_ave = 0.13746106908
    drugcount_months = 0.125304891143
    labCount_max = 0.101091758852
    labCount_min = 0.0194693581889
    labCount_ave = 0.0652144810954
    labcount_months = 0.130701721624
    labNull = -0.0658304784586
    drugNull = -0.0667253925286

    """
def TestUserInY2():
    """
    第二年有住院的特性
    """
    pass


import pandas as pd
dih=pd.read_csv("DayInHospital_Y2.csv")
c=pd.read_csv("Claims_Y1.csv")
dih['predict'] = pd.Series([0]*shape[0], index=dih.index)
dih['predict'] = pd.Series([0]*dih.shape[0], index=dih.index)
dih['claims_count'] = pd.Series([0]*dih.shape[0], index=dih.index)
for i in dih.memberid:
    dih.claims_count[dih.memberid==i]=c[c.MemberID==i].count()

