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

def getT(t, e, f): # t[e][f]
	return IBMModel1.getT(t, e, f)

class LMTranslator:
	def __init__(self, t, corpus, eLM, dict1 = None, dict2 = None):
		"""Initialize the data structures in the constructor."""
		self.t = t
		self.corpus = corpus
		self.eLM = eLM
		# parameters
		self.nwords = 5
		self.use_tp = True
		self.use_unigram = False
		self.use_bigram = False
		# build dictionary
		self.fWordsNotFound = []
		if dict1 is None or dict2 is None:
			self.buildDictionary()
			Utility.writeDictionary(self.dict1, '../output/LMT.dict1')
			Utility.writeDictionary(self.dict2, '../output/LMT.dict2')
		else:
			self.dict1 = Utility.readDictionary(dict1)
			self.dict2 = Utility.readDictionary(dict2)
		Utility.outputObject('../output/foreign_words_not_found', self.fWordsNotFound)

	def translateSentence(self, fText):
		""" Pick the best translation based on the probabilities calculated from the EM algorithm
			and the language model (trigram)."""
		eText, fText = [], fText.split(' ')
		prev1, prev2 = '<S>', '<S>'
		for i, ftoken in enumerate(fText):
			lftoken = ftoken.lower()
			if IBMModel1.toRemove(lftoken):
				if re.findall(r'\d+\,\d+', ftoken): # replace decimal point
					ftoken = ftoken.replace(',', '.')
				eText.append([(ftoken, float(1))]) # must be in list format
			else:
				if lftoken in self.dict1:
					etoken1 = self.dict1[lftoken]
					etoken2 = self.dict2[lftoken]
				else:
					etoken1, etoken2 = self.translateWord(lftoken)
				# if the fWord points to NULL, jump to next word
				if etoken1 == 'NULL':
				    continue
				# get the translation probability for eToken1 and eToken2
				us1 = self.eLM.scoreUnigram(etoken1) if self.use_unigram else float(1)
				tp1 = getT(self.t, etoken1, ftoken) if self.use_tp else float(1)
				etoken1 = (etoken1, tp1 * us1)
				if etoken1 != etoken2 and etoken2 != 'NULL':
					us2 = self.eLM.scoreUnigram(etoken2) if self.use_unigram else float(1)
					tp2 = getT(self.t, etoken2, ftoken) if self.use_tp else float(1)
					etoken2 = (etoken2, tp2 * us2)
					eText.append([etoken1, etoken2])
				else:
					 eText.append([etoken1])
		
		# reduce permutation complexity
		if not self.use_bigram:
			eText = self.reduceSentencePossibilities(eText)
		else:
			for i in xrange(1, len(eText)):
				beste1, bestScore = None, float(-1)
				for e1 in eText[i]:
					(word1, score1) = e1
					for e2 in eText[i - 1]:
						(word2, score2) = e2
						score = self.eLM.scoreBigram(word1, word2)
						if score > bestScore:
							score, beste1 = bestScore, e1
				eText[i] = [e1]
		
		# find the best permutation
		permutation, score = self.scorePermutations([], eText)
		etokens = [etoken for (etoken, tprob) in permutation]
		# final translation fixup
		etokens = Utility.finalTranslationTokensFixup(etokens)
		esentence = ' '.join(etokens)
		esentence = Utility.finalTranslationStringFixup(esentence)
		return esentence, score

	def reduceSentencePossibilities(self, eText): # eText = [(etoken, tprob)...]
		nwords = self.nwords if len(eText) < 50 else 1
		lst = [etokens[1] for etokens in eText if len(etokens) >= 2] # get a list of etoken2
		#lst = [(etoken2, len(etoken2)) for (etoken2, tprob2) in lst] # etoken2 = (etoken2, tprob2)
		lst.sort(key = lambda etoken2: etoken2[1], reverse = True)
		best_etokens = [etoken2 for (etoken2, len2) in lst[:nwords]]
		result = [etokens if (len(etokens) == 1 or etokens[1][0] in best_etokens) else etokens[:1] for etokens in eText]
		return result

	def scorePermutations(self, partial, remaining): # eText = [(etoken, tprob)...]
		# check for termination criteria
		if not remaining:
			plist = [tprob for (etoken, tprob) in partial]
			score = float(1)
			for p in plist:
				score *= p
			score *= self.eLM.scoreSentence(partial)
			return partial, score
		
		# find the best permuation
		bestPermutation, bestScore = None, float(-1)
		first, rest = remaining[0], remaining[1:]
		for word in first:
			permutation, score = self.scorePermutations(partial + [word], rest)
			if score > bestScore:
				bestPermutation, bestScore = permutation, score
		
		return bestPermutation, bestScore

	def buildDictionary(self):
		"""Build two dictionaries: one for the best match and the second for the second best match."""
		fWords = set()
		for (eLine, fLine) in self.corpus:
			fWords |= set(fLine)
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
	def __init__(self, type, ngram = 'trigram'):
		"""Initialize the data structures in the constructor."""
		self.type = type
		self.ngram = ngram
		assert(ngram == 'trigram' or ngram == 'bigram')
		self.trigramCounts = defaultdict(lambda: 0)
		self.bigramCounts = defaultdict(lambda: 0)
		self.unigramCounts = defaultdict(lambda: 1)
		self.unigramTotal = 0
		self.logdiscount = math.log(0.4)

	def train(self, corpus):
		""" Takes a corpus and trains the language model."""  
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
		""" Takes a list of strings as argument and returns the probability"""
		score = 0.0 
		tokens = ['<S>'] + sentence
		if self.ngram == 'trigram':
			for i in xrange(1, len(tokens)):
				word1 = tokens[i]
				word2 = tokens[i - 1]
				word3 = tokens[i - 2] if i >= 2 else tokens[0]
				score += self.scoreTrigramLog(word1, word2, word3)
		else: # bigram
			for i in xrange(1, len(tokens)):
				word1 = tokens[i]
				word2 = tokens[i - 1]
				score += self.scoreBigramLog(word1, word2)
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

	def scoreBigramLog(self, word1, word2):
		count2 = self.bigramCounts[(word1,word2)]
		count1 = self.unigramCounts[word1]
		if count2 > 0 and count1 > 0:
			score = math.log(count2) - math.log(count1)
		else:
			score = self.logdiscount
			score += math.log(count1) - math.log(self.unigramTotal)
		return score

	def scoreBigram(self, word1, word2):
		return math.exp(self.scoreBigramLog(word1, word2))

	def scoreUnigram(self, word):
		return float(self.unigramCounts[word]) / float(self.unigramTotal)


def main():
	if len(sys.argv) < 6:
		print "Invoke the program with: python baseline.py eTrain fTrain nIteration fTest output [tfile] [dict1 dict2]"
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
	eLM = TrigramLanguageModel(type = 'e', ngram = 'trigram')
	#eLM = TrigramLanguageModel(type = 'e', ngram = 'bigram')
	eLM.train(corpus)

	if len(sys.argv) > 8:
		print 'Reading dictionary...'
		LMT = LMTranslator(t, corpus, eLM, dict1 = sys.argv[7], dict2 = sys.argv[8])
	else:
		print 'creating dictionary...'
		LMT = LMTranslator(t, corpus, eLM)

	print 'Translating...'
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	outlm = open('../output/translations.lm', 'w+')
	for fLine in fTest:
		eLine, _ = LMT.translateSentence(fLine)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
		print >>outlm, fLine
		print >>outlm, eLine
		print >>outlm
	fTest.close()
	output.close()
	outlm.close()


if __name__ == "__main__":
	main()
