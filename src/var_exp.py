#from reg_mul import *
import pandas as pd
import numpy as np
from regression import readFile

l = ['ar','ca','de','es','fa','ga','hr','it','ko','no','ro','sme','uk','zh','bg','cs','el','et','fi','he','hu','ja','lv','pl','ru','sv','ur','bxr','da','en','eu','fr','hi','id','kmr','nl','pt','sl','tr','vi']


def getDistGouv(langue):
	lines = readFile(langue)
	res = []
	for l in lines:
		if isinstance(l[0], float) or isinstance(l[0], int) or l[0].isdigit() :
			if isinstance(l[6], float) or isinstance(l[6], int) or l[6].isdigit():
				res.append(int(l[0])-int(l[6]))
	return np.array(res)

def getMeanDist(langue):
	dists = getDistGouv(langue)
	return np.mean(np.abs(dists))

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

def mean_phrase_len(langue):
	lines = readFile(langue)
	phrases = np.array(reduceToPhrases(lines))
	lengths= []
	for phrase in phrases:
		lengths.append(len(phrase))
	return np.mean(lengths)

#def getCrossDependencyCount(langue):
#	lines = readFile(langue)

#getCrossDependencyCount('fr')
for lang in l:
	print(lang,end=':')
	print(mean_phrase_len(lang))
#	print(getMeanDist(lang))
