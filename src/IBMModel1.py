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

import sys
import decimal
from decimal import Decimal
import collections
from copy import deepcopy
from math import pow
from collections import defaultdict

decimal.getcontext().prec = 4
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

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
	punct = [",", ".", "/", "?", "|", "\\", "%", "$", "^", "&", "*", "(", ")", "-", ";", "&quot;", "&apos;s", "&#93;", "...", "&apos;t", ".\n", "&quot;\n", ":", "!", "¿"]
	stopWords = ["el", "a", "la", "los", "las", "de", "en", "un", "una", "y", "con", "del", "que", "por", "se"]
	tokens = []
	for i in range(len(line)):
		token = line[i].strip()
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
	for fWord in fWords:
		bestWord = ""
		maxProb = Decimal(-1)
		for eWord in t:
			if t[eWord][fWord] > maxProb:
				bestWord = eWord
				maxProb = t[eWord][fWord]
		if maxProb > Decimal(1) / Decimal(len(fWords)):
			print fWord + "<-" + bestWord + ": " + str(maxProb)

def train(corpus, nIt):
	""" This function performs the EM algorithm on the given corpus! For the E step, 
		since it is unnecesary to store all of the alignments in memory (also inefficient),
		we generate alignments for every sentence pair on the fly and add normalized probabilities
		to corresponding word pairs after examining each alignment. For the M step, we
		just normalize the probabilities for each foreign word """
	# Initialization: first, find the vocab of each language
	fWords = set()
	eWords = set()
	print "Putting into corpus..."
	for (eLine, fLine) in corpus:
		for eWord in eLine:
			eWords.add(eWord)
		for fWord in fLine:
			fWords.add(fWord)
	# t is the eventual translation probability matrix to be returned, while runningT is
	# used in the loop to calculate new probabilities
	print "Initializing translation probs..."
	# initialize all probabilities p(f | e) to be the same as 1/len(fWords)
	runningT = {}
	t = {}
	for eWord in eWords:
		runningT[eWord] = defaultdict(Decimal)
		t[eWord] = defaultdict(lambda: Decimal(1) / Decimal(len(fWords)))

	# Loop component: performs E and M steps in each iteration, until iteration times out
	for it in range(nIt):
		print "ITERATION " + str(it)
		printT(t, fWords)
		# E step: goes through each sentence pair, and examine the alignments
		for (eLine, fLine) in corpus:
			m = len(fLine)
			l = len(eLine)
			normConst = Decimal(1) / Decimal(pow(m, (l+1)))
			# Use the equation that 
			# p(fj, A | ei) = \frac{1}{(l+1)^m} * t(fj | ei) * \prod_{j' \neq j}^m\sum_{i'=1}^l * t(f_j' | e_i')
			sumProbs = []
			for j in range(m):
				sum = Decimal(0)
				for i in range(l):
					sum += Decimal(t[eLine[i]][fLine[j]])
				sumProbs.append(sum)
			for i in range(l):
				for j in range(m):
					fj = fLine[j]
					ei = eLine[i]
					# probability of p(f_j | e_i) provided by all alignments in this sentence pair
					prob = Decimal(1)
					for jp in range(m):
						if jp != j:
							prob *= sumProbs[jp]
					prob *= t[ei][fj] * normConst
					runningT[ei][fj] += prob
		# M step: since we have already accrued the probabilities from different alignments
		# in the E step, we only need to normalize the total probability of one eWord to be 1 here
		for eWord in runningT:
			wordSum = Decimal(0)
			for fWord in runningT[eWord]:
				wordSum += runningT[eWord][fWord]
			#if wordSum <= Decimal(0.00000000000001):
			#	continue
			for fWord in runningT[eWord]:
				runningT[eWord][fWord] /= wordSum
		t = deepcopy(runningT)
		for eWord in runningT:
			runningT[eWord] = defaultdict(Decimal)

		print
	printT(t, fWords)

	return t