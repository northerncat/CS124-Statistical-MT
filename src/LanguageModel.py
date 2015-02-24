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
from decimal import Decimal

class LMTranslator:
	def __init__(self, t, corpus, build = True):
		"""Initialize the data structures in the constructor."""
		self.t = t
		self.corpus = corpus
		self.eLM = TrigramLanguageModel('e')
		self.fLM = TrigramLanguageModel('f')
		self.eLM.train(corpus)
		self.fLM.train(corpus)
		if build:
			self.fWordsNotFound = []
			self.buildDictionary()
			outputObject('../output/foreign_words_not_found.txt', self.fWordsNotFound)
		else:
			self.readDictionary()

	def translate(self, fText):
		""" Pick the best translation based on the probabilities calculated from the EM algorithm
			and the language model (trigram)."""
		eText = []
		for i, token in enumerate(fText.split(' ')):
			ltoken = token.lower()
			if IBMModel1.toRemove(ltoken):
				if re.findall(r'\d+\,\d+', token): # replace decimal point
					token = token.replace(',', '.')
				eText.append([token]) # must be in list format
			else:
				if ltoken in self.dict1:
					eToken1, eToken2 = self.dict1[ltoken], self.dict2[ltoken]
				else:
					continue # eToken1, eToken2 = self.translateWord(ltoken)
				# if the fWord points to NULL, jump to next word
				if eToken1 == 'NULL' or eToken2 == 'NULL': continue
				eText.append([eToken1] if eToken1 == eToken2 else [eToken1, eToken2])
		permutation, score = self.scorePermutations([], eText)
		return " ".join(permutation)

	def scorePermutations(self, partial, remaining):
		if not remaining:
			return partial, self.eLM.score(partial)
		bestPermutation, bestScore = None, float(-1)
		first, rest = remaining[0], remaining[1:]
		for word in first:
			permutation, score = self.scorePermutations(partial + [word], rest)
			if score > bestScore:
				bestPermutation, bestScore = permutation, score
		return bestPermutation, bestScore

	def readDictionary(self, infile1 = '../output/dict1.txt', infile2 = '../output/dict2.txt'):
		self.dict1 = inputObject(infile1)
		self.dict2 = inputObject(infile2)

	def buildDictionary(self, outfile1 = '../output/dict1.txt', outfile2 = '../output/dict2.txt'):
		fWords = set()
		for (eLine, fLine) in self.corpus:
			fWords |= set(fLine) # union: fWords = fWords | set(fLine)
		IBMModel1.default_t_value = Decimal(1) / Decimal(len(fWords))
		self.dict1, self.dict2 = {}, {}
		ekeys = self.t.keys()
		for f in fWords:
			self.dict1[f], self.dict2[f] = self.translateWord(ekeys, f)
		#outputObject(outfile1, self.dict1)
		#outputObject(outfile2, self.dict2)

	def translateWord(self, ekeys, f):
		""" This function searches the translation probability dictionary to find an
			e word that maximizes p(token | eWord). """
		bestWord1, bestWord2 = 'NULL', 'NULL'
		bestProb1, bestProb2 = float(-1), float(-1)
		for e in ekeys:
			if f in self.t[e]:
				val = float(self.t[e][f])
				if val > bestProb1:
					bestProb1, bestWord1 = val, e
				val *= self.eLM.scoreUnigram(e)
				if val > bestProb2:
					bestProb2, bestWord2 = val, e
		if bestWord1 == 'NULL':
			self.fWordsNotFound.append(f)
		return bestWord1, bestWord2


class TrigramLanguageModel:
	def __init__(self, type):
		"""Initialize the data structures in the constructor."""
		self.type = type
		self.trigramCounts = collections.defaultdict(lambda: 0)
		self.bigramCounts = collections.defaultdict(lambda: 0)
		self.unigramCounts = collections.defaultdict(lambda: 1)
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

	def score(self, sentence):
		""" Takes a list of strings as argument and returns the log-probability of the 
			sentence using your language model. Use whatever data you computed in train() here. """
		score = 0.0 
		tokens = ['<S>'] + sentence
		for i in xrange(1, len(tokens)):
			word1 = tokens[i]
			word2 = tokens[i - 1]
			word3 = tokens[i - 2] if i >= 2 else tokens[0]
			count3 = self.trigramCounts[(word1,word2,word3)]
			count2 = self.bigramCounts[(word1,word2)]
			if count3 > 0 and count2 > 0:
				score += math.log(count3) - math.log(count2)
			else:
				score += self.logdiscount
				count1 = self.unigramCounts[word1]
				if count2 > 0 and count1 > 0:
					score += math.log(count2) - math.log(count1)
				else:
					score += self.logdiscount
					score += math.log(count1) - math.log(self.unigramTotal)
		return math.exp(score)

	def scoreUnigram(self, word):
		return float(self.unigramCounts[word]) / float(self.unigramTotal)


def outputObject(outfile, dict_list): # causing all strings to become unicode
	file = open(outfile, 'w')
	json.dump(dict_list, file, ensure_ascii=True, encoding="utf-8")
	file.close()

def inputObject(infile): # causing all strings to become unicode
	file = open(infile)
	dict_list = json.load(file, encoding="utf-8")
	file.close()
	return dict_list


def main_legacy():
	#eFile, fFile = '../data/train/europarl-v7.es-en.en', '../data/train/europarl-v7.es-en.es'
	eFile, fFile = '../data/dev/newstest2012.en', '../data/dev/newstest2012.es'
	corpus = IBMModel1.readCorpus(eFile, fFile)
	t = IBMModel1.readWholeT('../output/devT')
	LMT = LMTranslator(t, corpus, build = True)
	fTest = open('ftest', 'r')
	output = open('../output/ftest.en', 'w+')
	for line in fTest:
		eLine = LMT.translate(line)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()
	return

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
		t = IBMModel1.train(corpus, nIt)
	print 'Creating language model...'
	LMT = LMTranslator(t, corpus, build = True)
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	print 'Translating...'
	for line in fTest:
		eLine = LMT.translate(line)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()


if __name__ == "__main__":
	main()
