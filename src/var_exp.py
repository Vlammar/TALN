#from reg_mul import *
import pandas as pd
import numpy as np


langues = ['ar','ca','es','fa','ga','hr','it','ko','no','ro','uk','zh','bg','cs','el','et','fi','he','hu','ja','lv','pl','ru','sv','ur','da','en','eu','fr','hi','id','nl','pt','sl','tr','vi']
POS = ['X', 'PUNCT', 'NOUN', 'ADJ', 'VERB', 'NUM', '_', 'ADP', 'PRON', 'CCONJ', 'AUX', 'DET', 'ADV', 'PART', 'PROPN', 'SYM', 'SCONJ', 'INTJ']

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
		#print(word[1])
	return np.mean(words)
def getMeanLemmaLength(lines):
	words=[]
	for word in lines:
		words.append(len(str(word[2])))
	   # print(word[1])
	return np.mean(words)


def reduceToPhrases(lines):
	phrases = []
	current_phrase = []
	for l in lines:
		if not isinstance(l[0], float) and not isinstance(l[0], int) and not l[0].isdigit() :
			continue
		newline = len(current_phrase) != 0 and int(l[0]) <= int(current_phrase[-1][0])
		if newline:
			phrases.append(current_phrase)
			current_phrase = []
		current_phrase.append(l)
	return phrases

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


#def getCrossDependencyCount(langue):
#	lines = readFile(langue)
#getCrossDependencyCount('fr')
