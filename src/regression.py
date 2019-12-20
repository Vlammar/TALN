#regression


import numpy as np
import matplotlib.pyplot as lt
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from var_exp import *

def readFile(langue):
    datas = pd.read_csv("../corpus_equilibre/"+langue+"/"+langue+"_test.conllu",
    sep='\t',
    lineterminator='\n',
    index_col=False,quotechar=None, quoting=3,
                    names=["ID1","FORM2","LEMMA3","POS4","EMPTY5","MORPHO6","GOV7","LABEL8", "EMPTY9","EMPTY0","LANG"])
    #print(datas.columns)
    return datas.to_numpy()


def loadExplicativeVariable(path):
    X=[]

    for lg in langues:
        xlg=[]
        #print(readFile(lg)[:10])

        r=readFile(lg)

        #Features:
        #Longueur moyenne de la chaine de dependance
        meanDist=getMeanDist(r)
        xlg.append(meanDist)
        xlg.append(np.log(meanDist))

        meanPhraseLen=mean_phrase_len(r)
        xlg.append(meanPhraseLen)
        xlg.append(np.log(meanPhraseLen))

        MeanWordLength=getMeanWordLength(r)
        xlg.append(MeanWordLength)
        xlg.append(np.log(MeanWordLength))

        MeanLemmaLength=getMeanLemmaLength(r)
        xlg.append(MeanLemmaLength)
        xlg.append(np.log(MeanLemmaLength))

        wordUsed=nbWordUsed(r)
        xlg.append(wordUsed)
        xlg.append(np.log(wordUsed))

        lemmaUsed=nbLemmaUsed(r)
        xlg.append(lemmaUsed)
        xlg.append(np.log(lemmaUsed))

        charUsed=nbCharUsed(r)
        xlg.append(charUsed)
        xlg.append(np.log(charUsed))



        X.append(xlg)



    return np.array(X)
#INIT dictionnaire avec valeur donnee dans le sujet
y={}
y["hi"]=(79.47, 86.80)
y["it"]=(78.38, 82.15)
y["ur"]=(76.33, 83.55)
y["pl"]=(76.18, 84.41)
y["ja"]=(75.74, 85.60)
y["no"]=(73.25, 78.91)
y["bg"]=(73.40, 82.36)
y["el"]=(72.55, 78.52)
y["ca"]=(72.06, 79.70)
y["sv"]=(71.10, 77.36)
y["fr"]=(71.36, 77.02)
y["pt"]=(70.73, 76.95)
y["ru"]=(69.70, 73.85)
y["da"]=(68.12, 74.18)
y["id"]=(67.05, 72.21)
y["en"]=(67.18, 74.39)
y["es"]=(66.93, 74.52)
y["uk"]=(65.85, 74.19)
y["ro"]=(65.13, 72.53)
y["ga"]=(65.13, 74.02)
y["fa"]=(65.22, 73.42)
y["he"]=(64.68, 72.34)
y["et"]=(64.76, 75.40)
y["ar"]=(64.28, 71.65)
y["sl"]=(63.47, 71.78)
y["hr"]=(63.58, 72.10)
y["cs"]=(63.84, 72.45)
y["lv"]=(62.30, 69.83)
y["hu"]=(62.73, 68.86)
y["fi"]=(62.77, 70.83)
y["zh"]=(59.91, 65.15)
y["vi"]=(59.77, 62.68)
y["eu"]=(58.80, 68.78)
y["nl"]=(57.44, 68.43)
y["ko"]=(53.12, 63.21)
y["tr"]=(47.28, 55.20)


#construction des Y
Y_LAS=[]
Y_UAS=[]
for label,unlabel in y.values():
    Y_LAS.append(label)
    Y_UAS.append(unlabel)

#X=loadExplicativeVariable("../corpus_equilibre")
#print(X)
#reg = LinearRegression().fit(X, Y_LAS)
#print("LinearRegression score r2 {}".format(r2_score(Y_LAS, reg.predict(X))))
#print(*reg.coef_)
#print(reg.intercept_)

#import matplotlib.pyplot as plt

#plt.plot(np.arange(36),Y_LAS,'o')
#plt.plot(np.arange(36),reg.predict(X),'o')
#plt.legend(["true","pred"])
#plt.show()
print(langues)
