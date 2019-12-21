#regression


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from var_exp import *

def readFile(langue,tp):
	datas = pd.read_csv("../corpus_equilibre/"+langue+"/"+langue+"_"+tp+".conllu",
	sep='\t',
	lineterminator='\n',
	index_col=False,quotechar=None, quoting=3,
					names=["ID1","FORM2","LEMMA3","POS4","EMPTY5","MORPHO6","GOV7","LABEL8", "EMPTY9","EMPTY0","LANG"])
	#print(datas.columns)
	return datas.to_numpy()


def loadExplicativeVariable(path,tp):
	X=[]

	for lg in langues:
		xlg=[]

		r=readFile(lg,tp)
		names=[]
		#Features:
		#Longueur moyenne de la chaine de dependance
		meanDist=getMeanDist(r)
		xlg.append(meanDist)
		xlg.append(np.log(meanDist))

		names.append("meanDist")
		names.append("log meanDist")

		#Longueur moyenne des phrases
		meanPhraseLen=mean_phrase_len(r)
		xlg.append(meanPhraseLen)
		xlg.append(np.log(meanPhraseLen))

		names.append("mean_phrase_len")
		names.append("log mean_phrase_len")

		#Longueur moyenne des mots
		MeanWordLength=getMeanWordLength(r)
		xlg.append(MeanWordLength)
		xlg.append(np.log(MeanWordLength))

		names.append("getMeanWordLength")
		names.append("log getMeanWordLength")

		#Longueur moyenne des lemmes
		MeanLemmaLength=getMeanLemmaLength(r)
		xlg.append(MeanLemmaLength)
		xlg.append(np.log(MeanLemmaLength))

		names.append("getMeanLemmaLength")
		names.append("log getMeanLemmaLength")

		#Nombre de mots uniques utilises
		wordUsed=nbWordUsed(r)
		xlg.append(wordUsed)
		xlg.append(np.log(wordUsed))

		names.append("nbWordUsed")
		names.append("log nbWordUsed")

		#Nombre de lemmes uniques utilises
		lemmaUsed=nbLemmaUsed(r)
		xlg.append(lemmaUsed)
		xlg.append(np.log(lemmaUsed))

		names.append("nbLemmaUsed")
		names.append("log nbLemmaUsed")

		#Nombre de caractere uniques utilises
		charUsed=nbCharUsed(r)
		xlg.append(charUsed)
		xlg.append(np.log(charUsed))

		names.append("nbCharUsed")
		names.append("log nbCharUsed")

		#Mesure de l'ambiguite d'un part of speech
		#exemple: will peut etre un verbe et un nom
		POSambiguity=usePOSamb(r)
		xlg.append(POSambiguity)
		xlg.append(np.log(POSambiguity))

		names.append("usePOSamb")
		names.append("log usePOSamb")


		#Quartiles de la longueur des mots
		wQuartile=wordQuartile(r)

		for q in wQuartile:
			xlg.append(q)
			xlg.append(np.log(q))

		names.append("wordQuartile")
		names.append("log wordQuartile")

		#Quartiles de la longueur des lemmes
		lQuartile=lemmaQuartile(r)

		for q in lQuartile:
			xlg.append(q)
			xlg.append(np.log(q))

		names.append("lQuartile")
		names.append("log lQuartile")

		#Quartiles de la longueur des phrases
		sQuartile=sentenceQuartile(r)

		for s in sQuartile:
			xlg.append(s)
			xlg.append(np.log(s))

		names.append("sQuartile")
		names.append("log sQuartile")


		X.append(xlg)



	return np.array(X),np.array(names)
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


X,names=loadExplicativeVariable("../corpus_equilibre","test")


std_scaler = sklearn.preprocessing.StandardScaler()
X = std_scaler.fit_transform(X)
reg = Lasso().fit(X, Y_LAS)
print("LinearRegression score r2 {}".format(r2_score(Y_LAS, reg.predict(X))))
cpt=0
print("Variable explicative beta associe (ne donne pas l importance d une variable)")
for coef in reg.coef_[:len(names)]:

	print("{}  {} : {}".format(cpt,names[cpt],coef))
	cpt+=1

#print(*reg.coef_)
#print(reg.intercept_)

plt.plot(list(y.keys()),Y_LAS,'o')
plt.plot(list(y.keys()),reg.predict(X),'o')
plt.ylabel("LAS scores")
plt.xlabel("Languages")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.legend(["true","pred"])
plt.title("Comparison of prediction and real values of LAS")
plt.show()

nbvar=len(names)
deactivatedscore=[]
y_preds=[]
#mesure efficacite de chaque var en les retirants
#for var in range(nbvar):
#
#   Xdel=X
#   Xdel=np.delete(Xdel,var,1)
#	
#    reg = LinearRegression().fit(Xdel, Y_LAS)
#    print("On retire {}".format(names[var]))

#    print("LinearRegression score r2 {}".format(r2_score(Y_LAS, reg.predict(Xdel))))
#    deactivatedscore.append(r2_score(Y_LAS, reg.predict(Xdel)))
#    y_preds.append(reg.predict(Xdel))
#    cpt=0
   # print("Variable explicative beta associe (ne donne pas l importance d une variable)")
   # for coef in reg.coef_[:len(names)]:
    #    if(cpt==nbvar):
    #        continue
        #print("{}  {} : {}".format(cpt,names[cpt],coef))
     #   cpt+=1
#    print("\n\n")

#mesure pour chaque langue de la distance a la valeur reelle
y_preds=np.array(y_preds)

for i in range(nbvar):
    y_pred=y_preds[i]
    delta=np.absolute(Y_LAS-y_pred)

    plt.subplot(11,2,i+1)
    plt.plot(langues,delta,'o')
plt.show()

#Xtrain=loadExplicativeVariable("../corpus_equilibre","train")

#Xtest=loadExplicativeVariable("../corpus_equilibre","test")

#print(Xtrain)
#reg = LinearRegression().fit(Xtrain, Y_LAS)
#print("LinearRegression score r2 {}".format(r2_score(Y_LAS, reg.predict(Xtrain))))
#print("LinearRegression score r2 {}".format(r2_score(Y_UAS, reg.predict(Xtest))))
#print(*reg.coef_)
#print(reg.intercept_)

#plt.plot(np.arange(36),Y_LAS,'o')
#plt.plot(np.arange(36),reg.predict(Xtrain),'o')
#plt.legend(["true","pred"])
#plt.show()


#plt.plot(np.arange(36),Y_UAS,'o')
#plt.plot(np.arange(36),reg.predict(Xtest),'o')
#plt.legend(["true","pred"])
#plt.show()


