# LanguageModel.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This scripts builds the language models for English and Foreign languages.

import IBMModel1
import sys, math, collections, json, re
#from decimal import Decimal
from collections import defaultdict
import Utility

def getT(t, i, j): # t[i][j]
	return IBMModel1.getT(t, i, j)

class LMTranslator:
	def __init__(self, t, corpus, eLM, build = True):
		"""Initialize the data structures in the constructor."""
		self.t = t
		self.corpus = corpus
		self.eLM = eLM
		if build:
			self.fWordsNotFound = []
			self.buildDictionary()
			Utility.outputObject('../output/foreign_words_not_found.txt', self.fWordsNotFound)
		else:
			self.readDictionary()

	def translateSentence(self, fText):
		""" Pick the best translation based on the probabilities calculated from the EM algorithm
			and the language model (trigram)."""
		eText = []
		fText = fText.split(' ')
		prev1, prev2 = '<S>', '<S>'
		for i, token in enumerate(fText):
			ltoken = token.lower()
			if IBMModel1.toRemove(ltoken):
				if re.findall(r'\d+\,\d+', token): # replace decimal point
					token = token.replace(',', '.')
				eText.append([token]) # must be in list format
			else:
				if ltoken in self.dict1:
					eToken1, eToken2 = self.dict1[ltoken], self.dict2[ltoken]
				else:
					eToken1, eToken2 = self.translateWord(ltoken)
				# if the fWord points to NULL, jump to next word
				if eToken1 == 'NULL':
				    continue
				eText.append([eToken1] if eToken1 == eToken2 or eToken2 == 'NULL' else [eToken1, eToken2])
				#eText.append([eToken1])
		eText = self.reduceSentencePossibilities(eText)
		permutation, _ = self.scorePermutations([], eText)
		return " ".join(permutation)

	def reduceSentencePossibilities(self, eText):
		nwords = 1
		lst = [e[1] for e in eText if len(e) >= 2]
		lst = [(e, len(e)) for e in lst]
		lst.sort(key = lambda x: x[1], reverse = True)
		lst = [e[0] for e in lst[:nwords]]
		eText = [e if (len(e) == 1 or e[1] in lst) else e[:1] for e in eText]
		return eText

	def scorePermutations(self, partial, remaining):
		if not remaining:
			return partial, self.eLM.scoreSentence(partial)
		bestPermutation, bestScore = None, float(-1)
		first, rest = remaining[0], remaining[1:]
		for word in first:
			permutation, score = self.scorePermutations(partial + [word], rest)
			if score > bestScore:
				bestPermutation, bestScore = permutation, score
		return bestPermutation, bestScore

	def readDictionary(self, infile1 = '../output/dict1.txt', infile2 = '../output/dict2.txt'):
		self.dict1 = Utility.inputObject(infile1)
		self.dict2 = Utility.inputObject(infile2)

	def buildDictionary(self, outfile1 = '../output/dict1.txt', outfile2 = '../output/dict2.txt'):
		fWords = set()
		for (eLine, fLine) in self.corpus:
			fWords |= set(fLine) # union: fWords = fWords | set(fLine)
		self.dict1, self.dict2 = {}, {}
		for f in fWords:
			self.dict1[f], self.dict2[f] = self.translateWord(f)

	def translateWord(self, token):
		""" This function searches the translation probability dictionary to find an
			e word that maximizes p(token | eWord). """
		bestWord1, maxProb1 = 'NULL', float(-1)
		bestWord2, maxProb2 = 'NULL', float(-1)
		lst = [(e, getT(self.t, e, token)) for e in self.t if getT(self.t, e, token) > float(0)]
		lst.sort(key = lambda x: x[1], reverse = True)
		bestWord1 = lst[0][0] if len(lst) >= 1 else 'NULL'
		bestWord2 = lst[1][0] if len(lst) >= 2 else 'NULL'
		if bestWord1 == 'NULL':
			self.fWordsNotFound.append(token)
		return bestWord1, bestWord2


class TrigramLanguageModel:
	def __init__(self, type):
		"""Initialize the data structures in the constructor."""
		self.type = type
		self.trigramCounts = defaultdict(lambda: 0)
		self.bigramCounts = defaultdict(lambda: 0)
		self.unigramCounts = defaultdict(lambda: 1)
		self.unigramTotal = 0
		self.logdiscount = math.log(0.4)

	def train(self, corpus):
		""" Takes a corpus and trains your language model. 
			Compute any counts or other corpus statistics in this function. """  
		for tuple in corpus:
			tokens = tuple[0] if self.type == 'e' else tuple[1]
			tokens = ['<S>'] + tokens
			for i in xrange(1, len(tokens)):  
				word1 = tokens[i]
				word2 = tokens[i - 1]
				word3 = tokens[i - 2] if i >= 2 else tokens[0]
				self.trigramCounts[(word1,word2,word3)] += 1
				self.bigramCounts[(word1,word2)] += 1
				self.unigramCounts[word1] += 1
				self.unigramTotal += 1

	def scoreSentence(self, sentence):
		""" Takes a list of strings as argument and returns the log-probability of the 
			sentence using your language model. Use whatever data you computed in train() here. """
		score = 0.0 
		tokens = ['<S>'] + sentence
		for i in xrange(1, len(tokens)):
			word1 = tokens[i]
			word2 = tokens[i - 1]
			word3 = tokens[i - 2] if i >= 2 else tokens[0]
			score += self.scoreTrigramLog(word1, word2, word3)
		return math.exp(score)

	def scoreTrigramLog(self, word1, word2, word3):
		count3 = self.trigramCounts[(word1,word2,word3)]
		count2 = self.bigramCounts[(word1,word2)]
		if count3 > 0 and count2 > 0:
			score = math.log(count3) - math.log(count2)
		else:
			score = self.logdiscount
			count1 = self.unigramCounts[word1]
			if count2 > 0 and count1 > 0:
				score += math.log(count2) - math.log(count1)
			else:
				score += self.logdiscount
				score += math.log(count1) - math.log(self.unigramTotal)
		return score

	def scoreTrigram(self, word1, word2, word3):
		return math.exp(self.scoreTrigramLog(word1, word2, word3))

	def scoreUnigram(self, word):
		return float(self.unigramCounts[word]) / float(self.unigramTotal)


def main():
	if len(sys.argv) < 6:
		print "Invoke the program with: python baseline.py eTrain fTrain nIteration fTest output [tfile]"
		sys.exit()
	print 'Reading corpus...'
	eFile, fFile = sys.argv[1], sys.argv[2]
	corpus = IBMModel1.readCorpus(eFile, fFile)
	nIt = int(float(sys.argv[3]))
	if len(sys.argv) > 6:
		print 'Reading t matrix...'
		t = IBMModel1.readWholeT(sys.argv[6])
	else:
		print 'Training IBM Model 1...'
		t = IBMModel1.train(corpus, nIt, reverseCorpus = True, reverseT = True) # reverse has higher accuracy

	print 'Creating language model...'
	eLM = TrigramLanguageModel('e')
	eLM.train(corpus)

	LMT = LMTranslator(t, corpus, eLM, build = True)

	print 'Translating...'
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	for line in fTest:
		eLine = LMT.translateSentence(line)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()


if __name__ == "__main__":
	main()
