#regression


import numpy as np
import matplotlib.pyplot as lt
import sklearn
import pandas as pd

langues = ['ar','ca','de','es','fa','ga','hr','it','ko','no','ro','sme','uk','zh','bg','cs','el','et','fi','he','hu','ja','lv','pl','ru','sv','ur','bxr','da','en','eu','fr','hi','id','kmr','nl','pt','sl','tr','vi']

def readFile(langue):
    datas = pd.read_csv("../corpus_equilibre/"+langue+"/"+langue+"_test.conllu",
    sep='\t',
    lineterminator='\n',
    index_col=False,quotechar=None, quoting=3,
                    names=["ID1","FORM2","LEMMA3","POS4","EMPTY5","MORPHO6","GOV7","LABEL8", "EMPTY9","EMPTY0","LANG"])
    #print(datas.columns)
    return datas.to_numpy()

for lg in langues:
    print(readFile(lg)[:10])





def loadExplicativeVariable(path):
    X=[]

    for lg in langues:
        xlg=[]
        #print(readFile(lg)[:10])

        r=readFile(lg)
        xlg.append()

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


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
X=loadExplicativeVariable("../corpus_equilibre")
print(X)
clf = LogisticRegression(random_state=0).fit(X, Y_LAS)
print(r2_score(Y_LAS, clf.pred))
