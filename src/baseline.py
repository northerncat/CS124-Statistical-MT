# IBMModel1.py
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
from decimal import Decimal

def buildDictionary(t, corpus):
	fWords = set()
	for (eLine, fLine) in corpus:
		for fWord in fLine:
			fWords.add(fWord)
	dictionary = {}
	for fWord in fWords:
		bestWord = ""
		maxProb = Decimal(-1)
		for eWord in t:
			if t[eWord][fWord] > maxProb:
				bestWord = eWord
				maxProb = t[eWord][fWord]
		dictionary[fWord] = bestWord
	return dictionary

def translateWord(t, token):
	""" This function searches the translation probability dictionary to find an
		e word that maximizes p(token | eWord). """
	bestWord = ""
	maxProb = Decimal(-1)
	for eWord in t:
		if t[eWord][token] > maxProb:
			bestWord = eWord
			maxProb = t[eWord][token]
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
	for i in range(len(fText)):
		token = fText[i]
		if IBMModel1.toRemove(token.lower()):
			eText.append(token)
		else:
			if token.lower() in dictionary:
				eToken = dictionary[token.lower()]
			else:
				eToken = "NULL"
			# if the fWord points to NULL, jump to next word
			if eToken == "NULL":
				continue
			eText.append(eToken)
	eText = " ".join(eText)
	return eText

def main():
	if len(sys.argv) < 6:
		print "Invoke the program with: python baseline.py eTrain fTrain nIteration fTest outputDir"
		sys.exit()
	corpus = IBMModel1.readCorpus(sys.argv[1], sys.argv[2])
	nIt = int(float(sys.argv[3]))
	if len(sys.argv) > 6:
		t = IBMModel1.readWholeT(sys.argv[6])
	else:
		t = IBMModel1.train(corpus, nIt)
	dictionary = buildDictionary(t, corpus)
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	for line in fTest:
		eLine = baselineTranslate(t, dictionary, line)
		if eLine.endswith('\n'):
			output.write(eLine)
		else:
			output.write(eLine + '\n')
	fTest.close()
	output.close()

if __name__ == "__main__":
	main()
