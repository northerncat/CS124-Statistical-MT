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

def getT(t, i, j): # t[i][j]
	return IBMModel1.getT(t, i, j)

class Baseline:
	def __init__(self, t, corpus):
		"""Initialize the data structures in the constructor."""
		self.t = t
		self.corpus = corpus
		self.dictionary = self.buildDictionary()

	def buildDictionary(self):
		fWords = set()
		for (eLine, fLine) in self.corpus:
			fWords |= set(fLine)
		dictionary = {}
		for f in fWords:
			dictionary[f] = self.translateWord(f)
		return dictionary

	def translateWord(self, token):
		""" This function searches the translation probability dictionary to find an
			e word that maximizes p(token | eWord). """
		bestWord, maxProb = 'NULL', float(-1)
		for e in self.t:
			tef = getT(self.t, e, token)
			if tef > maxProb:
				bestWord, maxProb = e, tef
		return bestWord

	def translateSentence(self, fText):
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
				if ltoken in self.dictionary:
					eToken = self.dictionary[ltoken]
				else:
					eToken = self.translateWord(ltoken)
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
		t = IBMModel1.train(corpus, nIt, reverseCorpus = True, reverseT = True) # reverse has higher accuracy

	print 'Creating dictionary...'
	baseline = Baseline(t, corpus)

	print 'Translating...'
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	for fLine in fTest:
		eLine = baseline.translateSentence(fLine)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()

if __name__ == "__main__":
	main()
