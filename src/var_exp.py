#from reg_mul import *
import pandas as pd
import numpy as np


langues = ['ar','ca','es','fa','ga','hr','it','ko','no','ro','uk','zh','bg','cs','el','et','fi','he','hu','ja','lv','pl','ru','sv','ur','da','en','eu','fr','hi','id','nl','pt','sl','tr','vi']
POS = ['X', 'PUNCT', 'NOUN', 'ADJ', 'VERB', 'NUM', '_', 'ADP', 'PRON', 'CCONJ', 'AUX', 'DET', 'ADV', 'PART', 'PROPN', 'SYM', 'SCONJ', 'INTJ']
names={"ID":0,"FORM":1,"LEMMA":2,"POS":3,"EMPTY":4,"MORPHO":5,"GOV":6,"LABEL":7,"EMPTY_1":8,"EMPTY_2":9,"LANG":-1}
props=['Foreign', 'Case', 'Definite', 'Number', 'Gender', 'Aspect', 'Mood', 'Person', 'VerbForm', 'Voice', 'NumForm', 'AdpType', 'PronType', 'NumValue', 'Polarity', 'Abbr', 'NumType', 'Tense', 'PunctType', 'AdvType', 'Poss', 'PunctSide', 'Number[psor]', 'PrepCase', 'Polite', 'Reflex', 'Degree', 'Form', 'NounType', 'PartType', 'PrepForm', 'Dialect', 'Animacy', 'Gender[psor]', 'Clitic', 'Strength', 'Variant', 'Position', 'NameType', 'Style', 'Hyph', 'Animacy[gram]', 'ConjType', 'Connegative', 'Person[psor]', 'PartForm', 'InfForm', 'Derivation', 'Typo', 'HebBinyan', 'VerbType', 'HebSource', 'Prefix', 'HebExistential', 'Xtra', 'Number[psed]', 'Evident', 'Echo', 'Number[abs]', 'Person[abs]', 'Number[erg]', 'Person[erg]', 'Number[dat]', 'Person[dat]', 'Gender[erg]', 'Polite[erg]', 'Polite[abs]', 'Gender[dat]', 'Polite[dat]', 'Subcat']

#=====================================================================
#							UTILS
#=====================================================================

def getPhraseGov(phrase):
	gov = np.zeros(len(phrase)+1)
	for word in phrase:
		wordID = getNumber(word[names["ID"]])
		wordGOV = getNumber(word[names["GOV"]])
		if wordID is None or wordGOV is None:
			continue
		gov[wordID] = wordGOV
	return gov

def isANumber(s):
	return (isinstance(s, float) or isinstance(s, int) or(s.isdigit()))

def getNumber(s):
	if isANumber(s):
		return int(s)
	return None

def getAllDifferrentFeatures(lines,feature):
	feat={}
	for w in lines:
		feat[w[feature]]=1
	return feat.keys()

def POSambiguity(lines):
	words =  getAllDifferrentFeatures(lines,1)
	dic = {}
	for w in words:
		dic[w] = {}
	for l in lines:
		if l[3] in dic[l[1]]:
			dic[l[1]][l[3]] +=1
		else :
			dic[l[1]][l[3]] =1
	return dic


def reduceToPhrases(lines):
	phrases = []
	current_phrase = []
	for l in lines:
		if not isinstance(l[0], float) and not isinstance(l[0], int) and not l[0].isdigit() :
			continue
		newline = len(current_phrase) != 0 and int(l[0]) < int(current_phrase[-1][0])
		if newline:
			phrases.append(current_phrase)
			current_phrase = []
		current_phrase.append(l)
	return phrases

#=====================================================================
#							FEATURES
#=====================================================================


def getDistGouv(lines):
	res = []
	for l in lines:
		if isinstance(l[0], float) or isinstance(l[0], int) or l[0].isdigit() :
			if isinstance(l[6], float) or isinstance(l[6], int) or l[6].isdigit():
				res.append(int(l[0])-int(l[6]))
	return np.array(res)


def getMeanDist(lines):
	dists = getDistGouv(lines)
	return np.mean(np.abs(dists))

def getMeanWordLength(lines):
	words=[]
	for word in lines:
		words.append(len(str(word[1])))
	return np.mean(words)

def getMeanLemmaLength(lines):
	words=[]
	for word in lines:
		words.append(len(str(word[2])))
	return np.mean(words)

def mean_phrase_len(lines):
	phrases = np.array(reduceToPhrases(lines))
	lengths= []
	for phrase in phrases:
		lengths.append(len(phrase))
	return np.mean(lengths)

def nbWordUsed(lines):
	words={}
	for w in lines:
		words[w[1]]=1
	return len(words.keys())

def nbLemmaUsed(lines):
	words={}
	for w in lines:
		words[w[2]]=1
	return len(words.keys())

def nbCharUsed(lines):
	chars={}
	for w in lines:
		word=w[1]
		if (isinstance(w[1], float)or isinstance(w[1], int)or(w[1].isdigit())):
			word=str(w[1])
		for c in word:
			chars[c]=1
	return len(chars.keys())

def usePOSamb(lines):
	score=0
	d = POSambiguity(lines)
	for pos in d:
		if len(d[pos])>1:
			score += len(d[pos])
	return score/len(lines)

def wordQuartile(lines):
	words=[]
	for word in lines:
		words.append(len(str(word[1])))
	return [np.percentile(words, 25, axis=0),np.percentile(words, 50, axis=0),np.percentile(words, 75, axis=0)]

def lemmaQuartile(lines):
	words=[]
	for word in lines:
		words.append(len(str(word[2])))
	return [np.percentile(words, 25, axis=0),np.percentile(words, 50, axis=0),np.percentile(words, 75, axis=0)]

def sentenceQuartile(lines):
	phrases = np.array(reduceToPhrases(lines))
	lengths= []
	for phrase in phrases:
		lengths.append(len(phrase))
	return [np.percentile(lengths, 25, axis=0),np.percentile(lengths, 50, axis=0),np.percentile(lengths, 75, axis=0)]

def goUpLeaves(leaves,governors):
	paths = []
	for l in leaves:
		current_pos = l
		path = [current_pos]
		g = governors[l]
		while(g != 0):
			g = int(governors[current_pos])
			current_pos = g
			path.append(current_pos)
		paths.append(path)
	return paths

def getGovernerLinkLength(lines):
	phrases = np.array(reduceToPhrases(lines))
	phrase_path_length = []
	for phrase in phrases:
		govs = getPhraseGov(phrase)
		#on cherche toutes les feuilles : toutes celles pas présentes dans la table des gouvs
		leaves = [ i for i in range(1,len(govs)) if i not in govs]
		for path in goUpLeaves(leaves,govs):
			phrase_path_length.append(len(path))
	return np.mean(phrase_path_length)


def getCrossGov(lines):
	phrases = np.array(reduceToPhrases(lines))
	cross = 0
	for phrase in phrases:
		govs = getPhraseGov(phrase)
		for w1 in range(len(phrase)):
			for w2 in range(w1+1,len(phrase)):
				if govs[w1] < govs[w2]:
					cross +=1
	return cross

def getProperties(lines):
	dic={}
	for line in lines:
		s=line[5]
		if(s=="_"):
			continue
		types=s.split("|")
		for t in types:
			ls=t.split("=")
			dic[ls[0]]=1


	return list(dic.keys())

def getLanguageProp(lines):
	dic={}

	for prop in props:
		dic[prop]=[]

	for line in lines:
		s=line[5]
		if(s=="_"):
			continue
		types=s.split("|")
		for t in types:
			ls=t.split("=")
			if(not ls[1]in dic[ls[0]]):
				dic[ls[0]].append(ls[1])

	res=[]
	#print(dic)
	for i in range(len(props)):
		res.append(len(list(dic.values())[i]))
	#print(res)
	return res

def getPOSfreqLeaf():
	for p in POS:
		pass
