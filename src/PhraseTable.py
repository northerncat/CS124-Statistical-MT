# PhraseTable.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This script implements a phrase table based on the algorithm outlined in the 
# class slides, with the exception that the distortion decay factor is not implemented.

import IBMModel1
import sys, math, collections, json, re
#from decimal import Decimal
from collections import defaultdict
import Utility, LanguageModel

def getT(t, e, f): # t[e][f]
	return IBMModel1.getT(t, e, f)

default_phraseProbs_value = float(0)
def getP(pp, e, f): # self.phraseProbs[(e,f)]
	tpl = (e, f)
	return pp[tpl] if tpl in pp else default_phraseProbs_value

class PhraseTable:
	def __init__(self, t1, t2, corpus, eLM, LMT, ptFile = None):
		"""Initialize the data structures in the constructor."""
		self.t1 = t1 # t1[e][f] for t(f|e)
		self.t2 = t2 # t2[f][e] for t(e|f)
		self.corpus = Utility.removeNULLfromCorpus(corpus)
		self.eLM = eLM # trigram scoring for English
		self.LMT = LMT

		# parameters

		# phrase table: key = fphrase, value = ephrase
		self.phraseTable = defaultdict(lambda: '')

		# phrase probability: key = (ephrase, fphrase), value = probability
		self.phraseProbs = None

		# build or read phrase table
		if ptFile is None:
			self.buildPhraseTable()
			self.computePhraseProbabilities()
			self.writePhraseTables()
		else:
			self.readPhraseTables(ptFile)


	############ phrase table & probability table ###################


	def buildPhraseTable(self):
		"""Build a phrase table."""
		# calculate the default t value
		self.calculateDefaultTvalue()

		for (eLine, fLine) in self.corpus:
			# set up two tables for phrases for this pair of lines: pt1[e][f], pt2[e][f]
			pt1, pt2 = self.createInitialPhraseTables(eLine, fLine)

			# create an intersection and a union
			intersection, union = self.createIntersectionUnion(pt1, pt2, eLine, fLine)

			# use "consecutive" heuristics to pick some elements from union to intersection
			tbl = intersection
			tbl = self.applyHeuristicConsecutive(tbl, union, eLine, fLine)

			# use "language" heuristics to bundle words 
			tbl = self.applyHeuristicLanguage(tbl, union, eLine, fLine)

			# use "default" heuristic to use union to fill holes
			tbl = self.applyHeuristicDefault(tbl, union, eLine, fLine)

			# finally, extract phrases for this pair of lines
			self.extractPhrases(tbl, eLine, fLine)

		# use IBM Model 1 to fill the unigrams if not filled
		for f in self.LMT.dict1:
			e = self.LMT.dict1[f]
			if f in self.phraseTable:
				if 'NULL' in self.phraseTable[f]:
					self.phraseTable[f] = e
			else:
				self.phraseTable[f] = e

		# final fixup for NULL
		for fphrase in self.phraseTable:
			ephrase = self.phraseTable[fphrase]
			if 'NULL' in ephrase:
				if len(fphrase) > 3:
					self.phraseTable[fphrase] = fphrase # to cover unseen names
				else:
					self.phraseTable[fphrase] = '' # to avoid display NULL
		return

	def writePhraseTables(self, ptFile = '../output/phrase_table'):
		output = open(ptFile, 'w+')
		output.write('default_phraseProbs_value' + '\t' + 'default_phraseProbs_value' + '\t' + str(default_phraseProbs_value) + '\n')
		for fphrase in self.phraseTable:
			ephrase = self.phraseTable[fphrase]
			prob = self.phraseProbs[(ephrase, fphrase)]
			print >>output, fphrase + '\t' + ephrase + '\t' + str(prob)
		output.close()

	def readPhraseTables(self, ptFile):
		input = open(ptFile, 'r')
		firstLine = True
		for line in input:
			tokens = line.rstrip('\r\n').split('\t')
			if firstLine:
				global default_phraseProbs_value
				default_phraseProbs_value = float(tokens[2])
				self.phraseProbs = defaultdict(lambda: default_phraseProbs_value)
				firstLine = False

			fphrase, ephrase, prob = tokens[0], tokens[1], float(tokens[2])
			self.phraseTable[fphrase] = ephrase
			self.phraseProbs[(ephrase, fphrase)] = prob
		input.close()

	def calculateDefaultTvalue(self):
		eWords, fWords = set(), set()
		for (eLine, fLine) in self.corpus:
			eWords |= set(eLine)
			fWords |= set(fLine)
		IBMModel1.default_t_value = float(1) / (float(len(fWords)) * float(len(eWords)))

	def createInitialPhraseTables(self, eLine, fLine):
		"""Set up two inital tables for phrases for this pair of lines: pt1[e][f], pt2[e][f]."""
		fm, el = len(fLine), len(eLine)

		# initialize two phrase tables 
		pt1, pt2 = [], []
		for i in xrange(el):
			pt1.append([0 for j in xrange(fm)])
			pt2.append([0 for j in xrange(fm)])

		# using t1[e][f] to map f to best e
		for j, f in enumerate(fLine):
			ii, maxProb = -1, float(-1) # getT(self.t1, 'NULL', f)
			for i, e in enumerate(eLine):
				tef = getT(self.t1, e, f)
				if tef > maxProb:
					ii, maxProb = i, tef
			if ii > -1:
				pt1[ii][j] = 1

		# using t2[f][e] to map e to best f
		for i, e in enumerate(eLine):
			jj, maxProb = -1, float(-1) # getT(self.t2, 'NULL', e)
			for j, f in enumerate(fLine):
				tef = getT(self.t2, f, e)
				if tef > maxProb:
					jj, maxProb = j, tef
			if jj > -1:
				pt2[i][jj] = 1
		
		return pt1, pt2

	def createIntersectionUnion(self, pt1, pt2, eLine, fLine):
		"""Create an intersection and a union of two tables."""
		fm, el = len(fLine), len(eLine)
		intersection, union = [], []
		for i in xrange(el):
			r1, r2 = pt1[i], pt2[i] # two rows
			intersection.append([1 if r1[j] * r2[j] == 1 else 0 for j in xrange(fm)])
			union.append(       [1 if r1[j] + r2[j] > 0 else 0  for j in xrange(fm)])
		return intersection, union

	def applyHeuristicConsecutive(self, tbl, union, eLine, fLine):
		"""Use "consecutive" heuristics to pick some elements from union to intersection."""
		fm, el = len(fLine), len(eLine)
		for i in xrange(el):
			for j in xrange(1, fm):
				if union[i][j-1] * union[i][j] == 1: # intersection
					tbl[i][j-1], tbl[i][j] = 1, 1
		for j in xrange(fm):
			for i in xrange(1, el):
				if union[i-1][j] * union[i][j] == 1: # intersection
					tbl[i-1][j], tbl[i][j] = 1, 1
		return tbl

	def applyHeuristicLanguage(self, tbl, union, eLine, fLine):
		"""Use "language" heuristics to bundle words."""
		fm, el = len(fLine), len(eLine)
		bundle2 = [('a', 'la')]
		for (f1, f2) in bundle2:
			if f1 in fLine and f2 in fLine:
				j1, j2 = fLine.index(f1), fLine.index(f2)
				if j1 + 1 == j2: # union these two
					for i in xrange(el):
						u = 1 if tbl[i][j1] + tbl[i][j2] > 0 else 0 # union
						tbl[i][j1], tbl[i][j2] = u, u
		return tbl

	def applyHeuristicDefault(self, tbl, union, eLine, fLine):
		"""Use "default" heuristic to use union to fill holes."""
		fm, el = len(fLine), len(eLine)
		for j in xrange(fm):
			colsum = sum([tbl[i][j] for i in xrange(el)])
			if colsum == 0: # found a hole
				for i in xrange(el):
					tbl[i][j] = union[i][j] 
		return tbl

	def extractPhrases(self, tbl, eLine, fLine):
		"""Extract phrases from the phrase table."""
		fm, el = len(fLine), len(eLine)
		efgrams = []

		# extract fUnigram <-> eUnigram
		for i, e in enumerate(eLine):
			row = [tbl[i][_] for _ in xrange(fm)]
			if sum(row) == 1:
				j = row.index(1)
				col = [tbl[_][j] for _ in xrange(el)]
				if sum(col) == 1:
					efgrams.append(([e], fLine[j:j+1], i, j))

		# extract fNgram -> eUnigram
		for i, e in enumerate(eLine):
			j = 0
			while j < len(fLine) - 1: # capture longest possible combinations of consecutive 1's
				if tbl[i][j] * tbl[i][j+1] == 1: # found consecutive 1's
					jj = 2
					while j+jj < len(fLine) and tbl[i][j+jj] == 1:
						jj += 1
					# capture the fphrase, corresponding to consecutive 1's
					efgrams.append(([e], fLine[j:j+jj], i, j))
					j += jj
				else:
					j += 1
		
		# extract fUnigram -> eNgram
		for j, f in enumerate(fLine):
			i = 0
			while i < len(eLine) - 1: # capture longest possible combinations of consecutive 1's
				if tbl[i][j] * tbl[i+1][j] == 1: # found consecutive 1's
					ii = 2
					while i+ii < len(eLine) and tbl[i+ii][j] == 1:
						ii += 1
					# capture the ephrase, corresponding to consecutive 1's
					efgrams.append((eLine[i:i+ii], [f], i, j))
					i += ii
				else:
					i += 1

		# sort ascendingly by i value
		efgrams.sort(key = lambda x: x[2])

		# fix simple reverse, and sort efgrams again
		fix = self.fixReverse(efgrams)
		if fix:
			efgrams += fix
			efgrams.sort(key = lambda x: x[2])

		# combine phrases
		efgrams = self.combinePhrases([], efgrams)
		efgrams.sort(key = lambda x: x[2])

		# add translation matrix to the phrase table
		#for j, f in enumerate(fLine):
		#	if f in self.LMT.dict1:
		#		self.phraseTable[f] = self.LMT.dict1[f]

		# add these phrases to the phrase table
		for tpl in efgrams: # efgram = (ephrase, fphrase, i, j)
			ftokens = Utility.removeConsecutiveDuplicate(tpl[1])
			fphrase = ' '.join(ftokens)
			# add to phrase table only if fphrase has not been added
			if fphrase not in self.phraseTable:
				etokens = Utility.removeConsecutiveDuplicate(tpl[0])
				ephrase = ' '.join(etokens)
				self.phraseTable[fphrase] = ephrase # ptable[fphrase] = ephrase

		return # extractPhrases

	def fixReverse(self, efgrams):
		"""Fix simple reverse in the phrase table."""
		efgramsReverse = [] # efgram = (ephrase, fphrase, i, j)
		for k1 in xrange(len(efgrams)-1):
			for k2 in xrange(k1+1, len(efgrams)):
				(e1, f1, i1, j1) = efgrams[k1]
				(e2, f2, i2, j2) = efgrams[k2]
				if i1 + len(e1) == i2 and j1 - len(f2) == j2:
					efgramsReverse.append((e1 + e2, f2 + f1, i1, j2))
		return efgramsReverse

	def combinePhrases(self, partial, remaining):
		"""Combine phrases."""
		if not remaining:
			return partial
		first, rest = remaining[0], remaining[1:]
		combined = [first]
		for tpl in partial:
			(e1, f1, i1, j1) = tpl
			(e2, f2, i2, j2) = first
			if i1 + len(e1) == i2 and j1 + len(f1) == j2:
				combined.append((e1 + e2, f1 + f2, i1, j1))
		return self.combinePhrases(partial + combined, rest)

	def computePhraseProbabilities(self):
		"""Compute the probability of the phrases."""
		# collect counts
		total, countDict = 0, defaultdict(lambda: 1) # Laplace smoothing
		for (eLine, fLine) in self.corpus:
			eLine, fLine = ' '.join(eLine), ' '.join(fLine)
			for fphrase in self.phraseTable:
				ephrase = self.phraseTable[fphrase]
				if ephrase in eLine and fphrase in fLine:
					countDict[(ephrase, fphrase)] += 1
					total += 1

		vocabulary_len = len(countDict) # for Laplace smoothing
		laplace_total = float(total + vocabulary_len)

		global default_phraseProbs_value
		default_phraseProbs_value = float(1) / laplace_total

		self.phraseProbs = defaultdict(lambda: default_phraseProbs_value)

		# calculate probabilities based on the counts
		for fphrase in self.phraseTable:
			ephrase = self.phraseTable[fphrase]
			tpl = (ephrase, fphrase)
			self.phraseProbs[tpl] = float(countDict[tpl]) / laplace_total


	############ translation ###################


	def translateSentence(self, fText):
		"""Translate a f-sentence into an e-sentence."""
		# get LMT result
		#lmtResult, _ = self.LMT.translateSentence(fText)

		# initializations
		eText, fText = [], fText.split(' ')
		ssftokens = [] # subsentence ftokens
		for i, ftoken in enumerate(fText):
			lftoken = ftoken.lower()
			if IBMModel1.toRemove(lftoken):
				if ssftokens:
					eText += self.translateSubstence(ssftokens)
					ssftokens = [] # reset
				# do not translate this ftoken, just add it to eText
				if re.findall(r'\d+\,\d+', ftoken): # replace decimal point
					ftoken = ftoken.replace(',', '.')
				eText.append(ftoken)
			else:
				ssftokens.append(lftoken)
		
		# translate the last subsentence
		if ssftokens:
			eText += self.translateSubstence(ssftokens)
		# final fixup
		eText = Utility.removeConsecutiveDuplicate(eText)
		eText = Utility.finalTranslationTokensFixup(eText)
		ptResult = ' '.join(eText)
		ptResult = Utility.finalTranslationStringFixup(ptResult)
		return ptResult
		# pick the best of two results
		#ptScore = self.eLM.scoreSentence(ptResult.split(' '))
		#lmtScore = self.eLM.scoreSentence(lmtResult.split(' '))
		#return ptResult if ptScore >= lmtScore else lmtResult

	
	def translateSubstence(self, ssftokens):
		"""Translate a f-subsentence into an e-subsentence."""
		escores, etext = [], []
		
		# find all permutations
		permutations = self.findPermutations(ssftokens)

		# score each permutation
		for (prob, etokens) in permutations:
			score = self.eLM.scoreSentence(etokens)
			escores.append(score * prob)
			etext.append(etokens)

		# find the best one if there are results
		if escores and etext:
			indexofmax = escores.index(max(escores))
			etokens = etext[indexofmax]
			return Utility.removeNULLfromTokens(etokens)
		else:
			print 'NOTHING MATCH = ', permutations
			return [] # nothing match

	def findPermutations(self, ssftokens):
		"""Translate a f-subsentence into an e-subsentence."""
		# create a list of queues, one for each ssftokens index
		queues = [[] for _ in xrange(len(ssftokens))]

		# find all fphrases that are a substring of ssftokens
		for fphrase in self.phraseTable:
			ftokens = fphrase.split(' ')
			index = Utility.findSublist(ftokens, ssftokens)
			if index > -1:
				queues[index].append((fphrase, ftokens))

		# sort candidates ascendingly by their ftoken count
		queues = [sorted(q, key = lambda x: len(x[1])) if q else q for q in queues]

		# fill fword in empty queue
		#for i, q in enumerate(queues):
		#	if len(q) == 0:
		#		f = ssftokens[i]
		#		queues[i] = [(f, [f])]

		# find first nonzero queue
		first_nonzero_index = -1
		for i, q in enumerate(queues):
			if len(q) > 0:
				first_nonzero_index = i
				break

		# find all possible permutations
		permutations = []
		#self.findPermutationsRecursively([], 0, queues, permutations)
		self.findPermutationsRecursively([], first_nonzero_index, queues, permutations)

		# score the permutations, each permutation is now a tuple (prob, eSentence)
		return [self.scorePermutation(perm) for perm in permutations]


	def findPermutationsRecursively(self, partial, index, queues, permutations):
		# check for termination criteria
		if index >= len(queues):
			permutations.append(partial)
			return
		# for each token pair in the queue
		if queues[index]:
			for (fphrase, ftokens) in queues[index]:
				self.findPermutationsRecursively(partial + [ftokens], index + len(ftokens), queues, permutations)
		else:
			self.findPermutationsRecursively(partial, index + 1, queues, permutations)

	def scorePermutation(self, perm):
		"""Score a permutation and return a tuple (prob, etokens)."""
		log_prob, etokens = float(0), []
		for ftokens in perm:
			fphrase = ' '.join(ftokens)
			ephrase = self.phraseTable[fphrase] if fphrase in self.phraseTable else ''
			log_prob += math.log(getP(self.phraseProbs, ephrase, fphrase))
			etokens.append(ephrase)
		# combine all ephrases. recall that permutation's fphrases are sorted by index
		esentence = ' '.join(etokens)
		etokens = esentence.split(' ')
		return (math.exp(log_prob), etokens)


def main():
	if len(sys.argv) < 6:
		print "Invoke the program with: python baseline.py eTrain fTrain nIteration fTest output [tfile]"
		sys.exit()
	print 'Reading corpus...'
	eFile, fFile = sys.argv[1], sys.argv[2]
	corpus = IBMModel1.readCorpus(eFile, fFile)
	nIt = int(float(sys.argv[3]))

	if len(sys.argv) > 6: # t1[e][f]
		print 'Reading t1 matrix...'
		t1 = IBMModel1.readWholeT(sys.argv[6])
	else:
		print 'Training IBM Model 1 t1...'
		t1 = IBMModel1.train(corpus, nIt, reverseCorpus = True, reverseT = True, alt = False) # reverse has higher accuracy

	if len(sys.argv) > 9: # t2[f][e]
		print 'Reading t2 matrix...'
		t2 = IBMModel1.readWholeT(sys.argv[9])
	else:
		print 'Training IBM Model 1 t2...'
		t2 = IBMModel1.train(corpus, nIt, reverseCorpus = False, reverseT = True, alt = True) # reverse has higher accuracy

	print 'Creating language model...'
	eLM = LanguageModel.TrigramLanguageModel(type = 'e', ngram = 'trigram')
	eLM.train(corpus)

	if len(sys.argv) > 8: # LMT dict1, dict2
		print 'Reading language model dictionary..'
		LMT = LanguageModel.LMTranslator(t1, corpus, eLM, dict1 = sys.argv[7], dict2 = sys.argv[8])
	else:
		print 'Creating language model dictionary...'
		LMT = LanguageModel.LMTranslator(t1, corpus, eLM)

	if len(sys.argv) > 10: # Phrase Table
		print 'Reading t2 phrase table...'
		PT = PhraseTable(t1, t2, corpus, eLM, LMT, ptFile = sys.argv[10])
	else:
		print 'Creating phrase table...'
		PT = PhraseTable(t1, t2, corpus, eLM, LMT)

	print 'Translating...'
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	outpt = open('../output/translations.pt', 'w+')
	for fLine in fTest:
		eLine = PT.translateSentence(fLine)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
		print >>outpt, fLine
		print >>outpt, eLine
		print >>outpt
	fTest.close()
	output.close()
	outpt.close()

if __name__ == "__main__":
	main()
