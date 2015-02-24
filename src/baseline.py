# Baseline_EM.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This scripts performs the translation using only the probabilities obtained
# from IBM Model 1.

import IBMModel1
import sys
#from decimal import Decimal

def getT(t, e, f):
	return IBMModel1.getT(t, e, f)

def buildDictionary(t, corpus):
	fWords = set()
	for (eLine, fLine) in corpus:
		fWords |= set(fLine)
	dictionary = {}
	for f in fWords:
		dictionary[f] = translateWord(t, f)
	return dictionary

def translateWord(t, token):
	""" This function searches the translation probability dictionary to find an
		e word that maximizes p(token | eWord). """
	bestWord, maxProb = 'NULL', float(-1)
	for e in t:
		tef = getT(t, e, token)
		if tef > maxProb:
			bestWord, maxProb = e, tef
	return bestWord

def baselineTranslate(t, dictionary, fText):
	""" Using the probabilities calculated from the EM algorithm, this function
		performs translation on a piece of given foreign text. This is a baseline
		function so it only considers the alignment in which the two sentences are
		of same lengths, and each word aligns to the word at the same index. 

		Argument:
			t: the translation probability dictionary
			fText: a string in the source language (foreign) to be translated 
		Return:
			eText: a string translated to the target language (e) """
	eText = []
	fText = fText.split(" ")
	for i, token in enumerate(fText):
		ltoken = token.lower()
		if IBMModel1.toRemove(ltoken):
			eText.append(token)
		else:
			if ltoken in dictionary:
				eToken = dictionary[ltoken]
			else:
				eToken = translateWord(t, ltoken)
			# if the fWord points to NULL, jump to next word
			if eToken == "NULL":
				continue
			eText.append(eToken)
	eText = " ".join(eText)
	return eText

def main():
	if len(sys.argv) < 6:
		print "Invoke the program with: python baseline.py eTrain fTrain nIteration fTest output [tfile]"
		sys.exit()
	print 'Reading corpus...'
	corpus = IBMModel1.readCorpus(sys.argv[1], sys.argv[2])
	nIt = int(float(sys.argv[3]))
	if len(sys.argv) > 6:
		print 'Reading t matrix...'
		t = IBMModel1.readWholeT(sys.argv[6])
	else:
		print 'Training IBM Model 1...'
		t = IBMModel1.train(corpus, nIt, reverse = True) # reverse has higher accuracy
	print 'Creating dictionary...'
	dictionary = buildDictionary(t, corpus)
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	print 'Translating...'
	for line in fTest:
		eLine = baselineTranslate(t, dictionary, line)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()

if __name__ == "__main__":
	main()
