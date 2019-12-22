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

def plotHist(array):
	plt.bar(x=range(len(array)),height=np.array(array),width=.9,alpha=.8)

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

		names.append("POSambiguity")
		names.append("log POSambiguity")
		govLinkLen = getGovernerLinkLength(r)
		xlg.append(govLinkLen)
		xlg.append(np.log(govLinkLen))

		names.append("GovLinkLength")
		names.append("log GovLinkLength")
		crossGov= getCrossGov(r)
		xlg.append(crossGov)
		xlg.append(np.log(crossGov))


		names.append("CrossGov")
		names.append("log CrossGov")



		#Quartiles de la longueur des mots
		wQuartile=wordQuartile(r)

		for q in wQuartile:
			xlg.append(q)
			xlg.append(np.log(q))

		names.append("wordQuartile1")
		names.append("wordQuartile2")
		names.append("wordQuartile3")
		names.append("log wordQuartile1")
		names.append("log wordQuartile2")
		names.append("log wordQuartile3")

		#Quartiles de la longueur des lemmes
		lQuartile=lemmaQuartile(r)

		for q in lQuartile:
			xlg.append(q)
			xlg.append(np.log(q))

		names.append("lQuartile1")
		names.append("lQuartile2")
		names.append("lQuartile3")
		names.append("log lQuartile1")
		names.append("log lQuartile2")
		names.append("log lQuartile3")

		#Quartiles de la longueur des phrases
		sQuartile=sentenceQuartile(r)

		for s in sQuartile:
			xlg.append(s)
			xlg.append(np.log(s))

		names.append("sQuartile1")
		names.append("sQuartile2")
		names.append("sQuartile3")
		names.append("log sQuartile1")
		names.append("log sQuartile2")
		names.append("log sQuartile3")

		p=getLanguageProp(r)

		#Filter=[1,2,3,5,7,8,9]
		Filter=range(len(p))
		#Filter=[]
		for i in range(len(p)):
			if(i in Filter):
				xlg.append(p[i])
				names.append(props[i])
		X.append(xlg)



	return np.array(X),np.array(names)

def regression(X,y):

	std_scaler = sklearn.preprocessing.StandardScaler()
	Xreg = std_scaler.fit_transform(X)
	reg = Lasso(fit_intercept=True,alpha=0.5).fit(Xreg, y)
#	reg = LinearRegression().fit(X,y)
	print("Lasso score r2 {}".format(r2_score(y, reg.predict(Xreg))))

	return reg

def plotReg(reg,X,y,Y):
	plt.plot(list(y.keys()),Y,'o')
	std_scaler = sklearn.preprocessing.StandardScaler()
	Xreg = std_scaler.fit_transform(X)
	plt.plot(list(y.keys()),reg.predict(Xreg),'o')
	plt.ylabel("Scores")
	plt.xlabel("Languages")
	locs, labels = plt.xticks()
	plt.setp(labels, rotation=90)
	plt.legend(["true","pred"])
	plt.title("Comparison of prediction and real values of LAS")
	plt.show()




def computeAndSaveFeatures():
	X,features = X,names=loadExplicativeVariable("../corpus_equilibre","test")
	np.savetxt("X.in",np.array(X))

	with open("feats.in", 'w') as output:
		for row in features:
		    output.write(str(row) + '\n')

	return X,features,names

def loadFeatures():
	X=np.loadtxt("X.in")
	features = []
	with open("feats.in", 'r') as input_feats:
		line = input_feats.readline()
		while line :
			line = input_feats.readline()
			features.append(line)
	return X,features




def main():
	#INIT dictionnaire avec valeur donnee dans le sujet
	y={
		"hi":(79.47, 86.80),	"it":(78.38, 82.15),	"ur":(76.33, 83.55),	"pl":(76.18, 84.41),
		"ja":(75.74, 85.60),	"no":(73.25, 78.91),	"bg":(73.40, 82.36),	"el":(72.55, 78.52),
		"ca":(72.06, 79.70),	"sv":(71.10, 77.36),	"fr":(71.36, 77.02),	"pt":(70.73, 76.95),
		"ru":(69.70, 73.85),	"da":(68.12, 74.18),	"id":(67.05, 72.21),	"en":(67.18, 74.39),
		"es":(66.93, 74.52),	"uk":(65.85, 74.19),	"ro":(65.13, 72.53),	"ga":(65.13, 74.02),
		"fa":(65.22, 73.42),	"he":(64.68, 72.34),	"et":(64.76, 75.40),	"ar":(64.28, 71.65),
		"sl":(63.47, 71.78),	"hr":(63.58, 72.10),	"cs":(63.84, 72.45),	"lv":(62.30, 69.83),
		"hu":(62.73, 68.86),	"fi":(62.77, 70.83),	"zh":(59.91, 65.15),	"vi":(59.77, 62.68),
		"eu":(58.80, 68.78),	"nl":(57.44, 68.43),	"ko":(53.12, 63.21),	"tr":(47.28, 55.20)
	}


	#construction des Y
	Y_LAS = [y[score][0] for score in y]
	Y_UAS = [y[score][1] for score in y]

	X,f,names = computeAndSaveFeatures()
	#X,names = loadFeatures()

	#X,names=Xlin("../corpus_equilibre/","test")
	reg = regression(X,Y_LAS)
	print("Variable explicative beta associe ")
	print(reg.coef_)

	print("\nNom des variables utilises\n")
	for i in range(len(reg.coef_)):

		coef=reg.coef_[i]
		if(coef>0):
			print(names[i])
	plotReg(reg,X,y,Y_LAS)

	#nbvar=len(names)
	#deactivatedscore=[]
	#y_preds=[]
	#mesure efficacite de chaque var en les retirants
#	for var in range(nbvar):
#
#		Xdel=X
#		Xdel=np.delete(Xdel,var,1)
#
#		reg = LinearRegression().fit(Xdel, Y_LAS)
#		print("On retire {}".format(names[var]))
#
#		print("LinearRegression score r2 {}".format(r2_score(Y_LAS, reg.predict(Xdel))))
#		deactivatedscore.append(r2_score(Y_LAS, reg.predict(Xdel)))
#		y_preds.append(reg.predict(Xdel))
#		cpt=0
#		print("Variable explicative beta associe (ne donne pas l importance d une variable)")
#		for coef in reg.coef_[:len(names)]:
#			if(cpt==nbvar):
#				continue
#			print("{}  {} : {}".format(cpt,names[cpt],coef))
#			cpt+=1
#		print("\n\n")

	#mesure pour chaque langue de la distance a la valeur reelle
#	y_preds=np.array(y_preds)

#	for i in range(nbvar):
#		y_pred=y_preds[i]
#		delta=np.absolute(Y_LAS-y_pred)
#		plt.subplot(11,2,i+1)
#		plotHist(delta)
#		#plt.bar(range(len(langues)),delta,'o')
#	plt.show()

main()
