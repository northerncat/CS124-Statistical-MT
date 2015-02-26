# IBMModel1.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This module implements the Expectation-Maximization algorithm of IBM Model 1
# that computes the probability of translating certain foreign words from english
# words.
#
# Acknowledgment:
# The train() function that builds the IBM Model 1 translation probability matrix
# uses the algorithm published in the Statistical Machine Translation (SMT) site:
# http://www.statmt.org/book/slides/04-word-based-models.pdf
# However, our train() function computes t(f|e) rather than t(e|f) that
# the algorithm in the SMT site computes.

import sys
import decimal
#from decimal import Decimal
import collections
from copy import deepcopy
from math import pow
from collections import defaultdict

decimal.getcontext().prec = 4
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

def getT(t, i, j):
	return t[i][j] if j in t[i] else float(0)

def toRemove(string):
	""" This function checks if the given string is a number (floating point or integer)! """
	punct = [",", ".", "/", "?", "|", "\\", "%", "$", "^", "&", "*", "(", ")", "-", ";", "&quot;", "&apos;s", "&#93;", "...", "&apos;t", ".\n", "&quot;\n"]
	for char in string:
		if char.isdigit() or char in punct:
			return True
	return False

def removePunct(line, ef):
	""" This function removes the punctuation marks and numbers in a line represented by a list of tokens. """
	line = line.split(" ")
	punct = [",", ".", "/", "?", "|", "\\", "%", "$", "^", "&", "*", "(", ")", "-", ";", "&quot;", "&apos;s", "&#93;", "...", "&apos;t", ".\n", "&quot;\n", ":", "!", "\xc2"]
	stopWords = ["el", "a", "la", "los", "las", "de", "en", "un", "una", "y", "con", "del", "que", "por", "se"]
	tokens = []
	for i in xrange(len(line)):
		token = line[i].strip()
		if token == "":
			continue
		if not toRemove(token):
			if token[-1] == '\n':
				tokens.append(token[:-1].lower())
			else:
				tokens.append(token.lower())
	# give the foreign sentence an option to point to NULL
	if ef == 'e':
		tokens.append("NULL")
	return tokens

def readCorpus(eFile, fFile):
	""" This function reads in the two corpus files and builds the sentence-pair
		corpus for the EM algorithm. Note that this function requires that the 
		files are in the format given, i.e. a line in eFile corresponds to the 
		same index of line in fFile. The tokens consisting of only punctuation
		marks and/or numbers would be eliminated!

		Arguments: 
		eFile, fFile: string of the corpus files name. e represents source
						language, f represents target language

		Return:
			list(eS, fS): a list of tuples consisting of corresponding sentences.
							Each sentence is represented as a list of tokens! """

	eF = open(eFile, 'r')
	eLines = []
	for line in eF:
		tokens = removePunct(line, 'e')
		eLines.append(tokens)
	toReturn = []
	fF = open(fFile, 'r')
	i = 0
	for line in fF:
		tokens = removePunct(line, 'f')
		toReturn.append((eLines[i], tokens))
		i += 1
	return toReturn

def printT(t, fWords):
	for f in fWords:
		bestWord, maxProb = "", float(-1)
		for e in t:
			if getT(t, e, f) > maxProb:
				bestWord = e
				maxProb = getT(t, e, f)
		if maxProb > float(1) / float(len(fWords)):
			print f + "<-" + bestWord + ": " + str(maxProb)

def writeWholeT(t, alt = False):
	output = open("../output/wholeT.test" if not alt else "../output/wholeT2.test", "w+") ###############################
	for e in t:
		for f in t[e]:
			#print f + " " + e + " " + str(t[e][f])
			output.write(f + " " + e + " " + str(getT(t, e, f)) + "\n")
	output.close()

def readWholeT(filename):
	tFile = open(filename)
	t = {}
	for line in tFile:
		tokens = line.split(" ")
		f, e = tokens[0], tokens[1]
		prob = float(tokens[2])
		if e not in t:
			t[e] = defaultdict(lambda: float(0))
		t[e][f] = prob
	tFile.close()
	return t


def train(corpus, nIt, reverseCorpus = False, reverseT = False, alt = False):
	""" This function performs the EM algorithm on the given corpus! For the E step, 
		since it is unnecesary to store all of the alignments in memory (also inefficient),
		we generate alignments for every sentence pair on the fly and add normalized probabilities
		to corresponding word pairs after examining each alignment. For the M step, we
		just normalize the probabilities for each foreign word """
	# reverse the ordering of e and f
	if reverseCorpus:
		corpus2 = [(fLine, eLine) for (eLine, fLine) in corpus]
		corpus = corpus2

	# Initialization: first, find the vocab of each language
	eWords, fWords = set(), set()
	print "Putting into corpus..."
	for (eLine, fLine) in corpus:
		eWords |= set(eLine)
		fWords |= set(fLine)

	# t is the eventual translation probability matrix to be returned, while runningT is
	# used in the loop to calculate new probabilities

	# initialize all probabilities p(f | e) to be the same as 1/len(fWords)
	print "Initializing translation probs..."
	t = {}
	default_t_value = float(1) / float(len(fWords)) 
	for e in eWords:
		t[e] = defaultdict(lambda: default_t_value)
		
	# Loop component: performs E and M steps in each iteration, until iteration times out
	for it in xrange(nIt):
		print "ITERATION " + str(it)

		# E step

		# initialization
		runningT = {}
		for e in eWords:
			runningT[e] = defaultdict(lambda: float(0))
		sum = defaultdict(lambda: float(0))

		# goes through each sentence pair, and examine the alignments
		for (eLine, fLine) in corpus:
			fm, el = len(fLine), len(eLine)
			sumProbs = [float(0) for j in xrange(fm)]

			for i, e in enumerate(eLine):
				for j, f in enumerate(fLine):
					tef = t[e][f] if f in t[e] else default_t_value
					sumProbs[j] += tef

			for i, e in enumerate(eLine):
				for j, f in enumerate(fLine):
					tef = t[e][f] if f in t[e] else default_t_value
					runningT[e][f] += tef / sumProbs[j]
					sum[e] += tef / sumProbs[j]

		# M step
		
		# since we have already accrued the probabilities from different alignments
		# in the E step, we only need to normalize the total probability of one eWord to be 1 here
		for e in runningT:
			for f in runningT[e]:
				t[e][f] = runningT[e][f] / sum[e]

		# end of iteration

	# reverse the ordering of e and f
	if reverseT:
		# t[e][f] -> t2[f][e]
		default_t_value = float(1) / float(len(eWords))
		t2 = {}
		for f in fWords:
			t2[f] = defaultdict(lambda: default_t_value)
		for e in t:
			for f in t[e]:
				t2[f][e] = t[e][f]
		t = t2

	# output the t matrix
	writeWholeT(t, alt)
	return t


def main():
	corpus = readCorpus(sys.argv[1], sys.argv[2])
	nIt = int(float(sys.argv[3]))
	t = train(corpus, nIt)


if __name__ == "__main__":
	main()
