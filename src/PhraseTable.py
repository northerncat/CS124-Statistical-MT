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
import Utility, LanguageModel

def getT(t, i, j): # t[i][j]
	return IBMModel1.getT(t, i, j)

class PhraseTable:
	def __init__(self, t1, t2, corpus, eLM, LMT, build = True):
		"""Initialize the data structures in the constructor."""
		self.t1 = t1 # t1[e][f]
		self.t2 = t2 # t2[f][e]
		self.corpus = corpus
		self.eLM = eLM
		self.LMT = LMT
		self.phraseTable = defaultdict(lambda: [])
		self.phraseProbs = defaultdict(lambda: float(0))
		if build:
			self.buildPhraseTable()
		else:
			self.readPhraseTable()
		self.computePhraseProbabilities()

	def translateSentence(self, fText):
		lmt = self.LMT.translateSentence(fText)
		fText = fText.split(' ')
		eText = []
		tokens = []
		for i, token in enumerate(fText):
			ltoken = token.lower()
			if IBMModel1.toRemove(ltoken):
				if tokens:
					eText += self.translateSubstence(tokens)
					tokens = []
				eText.append(token)
			else:
				tokens.append(ltoken)
		if tokens:
			eText += self.translateSubstence(tokens)
		pt = ' '.join(eText)
		# choose the best from LMT and PT
		lmt_score = self.eLM.scoreSentence(lmt.split(' '))
		pt_score = self.eLM.scoreSentence(eText)
		return pt if pt_score >= lmt_score else lmt

	def translateSubstence(self, tokens):
		eScore, eText = [], []
		permutations = self.findPermutations(tokens)
		for (prob, sentence) in permutations:
			score = self.eLM.scoreSentence(sentence)
			eScore.append(score * prob)
			eText.append(sentence)
		if eScore and eText:
			indexofmax = eScore.index(max(eScore))
			sentence = eText[indexofmax]
			return ' '.join(sentence)
		else:
			return '' # nothing match

	def findPermutations(self, tokens):
		nperms = 10
		# find all substring fphrases
		candidates = []
		for fphrase in self.phraseTable:
			ftokens = fphrase.split(' ')
			index = Utility.findSublist(ftokens, tokens)
			if index > -1:
				candidates.append((index, fphrase, ftokens))
		# sort the phrases descendingly by their token count
		candidates.sort(key = lambda x: len(x[2]), reverse = True)
		# find possible permutations
		permutations = []
		for i in xrange(len(candidates)):
			perm = [candidates[i]]
			for j in xrange(i+1, len(candidates)):
				x1, y1 = candidates[j][0], candidates[j][0] + len(candidates[j][2]) - 1
				overlap = False
				for cand in perm:
					x2, y2 = cand[0], cand[0] + len(cand[2]) - 1
					if Utility.isRangeOverlapped(x1, y1, x2, y2): # check string indices overlapping
						overlap = True
						break
				if not overlap:
					perm.append(candidates[j])
			if sum([len(tpl[2]) for tpl in perm]) < len(tokens)-1: ###### only tolerate minus one in length
				continue
			perm.sort(key = lambda x: x[0])
			permutations.append(perm)
		# sort permutations based on their token count
		permutations.sort(key = lambda perm: sum([len(tpl[2]) for tpl in perm]), reverse = True)
		permutations = [self.scorePermutation(perm) for perm in permutations]
		return permutations if len(permutations) < nperms else permutations[:nperms]

	def scorePermutation(self, perm):
		prob = float(1)
		sentence = []
		for (index, fphrase, ftokens) in perm:
			ephrase = self.phraseTable[fphrase]
			prob *= self.phraseProbs[(ephrase, fphrase)]
			sentence.append(ephrase)
		sentence = ' '.join(sentence).split(' ')
		return (prob, sentence)
	
	def translate(self, partials, fText):
		newpartials = []
		for (j, prob, perm) in partials:
			added = []
			for j2 in xrange(j+1, len(fText)):
				flist = fText[j:j2]
				fphrase = ' '.join(flist)
				if fphrase in self.phraseTable:
					ephrase = self.phraseTable[fphrase]
					added.append((j2, prob * self.phraseProbs[(ephrase, fphrase)], perm + flist))
			newpartials += added if added else [(j, prob, perm)]
		return self.translate(newpartials, fText)


	def readPhraseTable(self, infile = '../output/PhraseTable.txt'):
		self.phraseTable = Utility.inputObject(infile)

	def buildPhraseTable(self, outfile = '../output/PhraseTable.txt'):
		for (eLine, fLine) in self.corpus:
			eLine = [e for e in eLine if e != 'NULL']
			fLine = [f for f in fLine if f != 'NULL']
			fm, el = len(fLine), len(eLine)

			# create two phrase tables for this pair of lines: pt1[e][f], pt2[e][f]
			pt1, pt2 = [], []
			for i in xrange(el):
				lst = [False for j in xrange(fm)]
				pt1.append(list(lst))
				pt2.append(lst)

			# using t1[e][f] to map f to best e
			for j, f in enumerate(fLine):
				ii, maxProb = -1, float(-1)
				for i, e in enumerate(eLine):
					tef = getT(self.t1, e, f)
					if tef > maxProb:
						ii, maxProb = i, tef
				if ii > -1:
					pt1[ii][j] = True

			# using t2[f][e] to map e to best f
			for i, e in enumerate(eLine):
				jj, maxProb = -1, float(-1)
				for j, f in enumerate(fLine):
					tef = getT(self.t2, f, e)
					if tef > maxProb:
						jj, maxProb = j, tef
				if jj > -1:
					pt2[i][jj] = True

			# create an intersection and a union
			intersection, union = self.createIntersectionUnion(pt1, pt2, eLine, fLine)

			# use "consecutive" heuristics to pick some elements from union to intersection
			tbl = self.applyHeuristicConsecutive(intersection, union, eLine, fLine)

			# use "language" heuristics to bundle words 
			tbl = self.applyHeuristicLanguage(tbl, union, eLine, fLine)

			# use "default" heuristic to use union to fill holes
			tbl = self.applyHeuristicDefault(tbl, union, eLine, fLine)

			# finally, extract phrases for this pair of lines
			self.extractPhrases(tbl, eLine, fLine)

		# output the phrase table for future use
		Utility.outputObject(outfile, self.phraseTable)

	# create intersection and union
	def createIntersectionUnion(self, pt1, pt2, eLine, fLine):
		fm, el = len(fLine), len(eLine)
		intersection, union = [], []
		for i in xrange(el):
			r1, r2 = pt1[i], pt2[i] # two rows
			intersection.append([r1[j] and r2[j] for j in xrange(fm)])
			union.append([r1[j] or r2[j] for j in xrange(fm)])
		return intersection, union

	# use "consecutive" heuristics to pick some elements from union to intersection
	def applyHeuristicConsecutive(self, tbl, union, eLine, fLine):
		fm, el = len(fLine), len(eLine)
		for i in xrange(el):
			for j in xrange(1, fm):
				if union[i][j-1] and union[i][j]:
					tbl[i][j-1], tbl[i][j] = True, True
		for j in xrange(fm):
			for i in xrange(1, el):
				if union[i-1][j] and union[i][j]:
					tbl[i-1][j], tbl[i][j] = True, True
		return tbl

	# use "language" heuristics to bundle words ######## ASK HELP ##########
	def applyHeuristicLanguage(self, tbl, union, eLine, fLine):
		fm, el = len(fLine), len(eLine)
		bundle2 = [('a', 'la')]
		for tpl in bundle2:
			if tpl[0] in fLine and tpl[1] in fLine:
				j1, j2 = fLine.index(tpl[0]), fLine.index(tpl[1])
				if j1 + 1 == j2: # union these two
					for i in xrange(el):
						u = tbl[i][j1] or tbl[i][j2]
						tbl[i][j1], tbl[i][j2] = u, u
		return tbl

	# use "default" heuristic to use union to fill holes
	def applyHeuristicDefault(self, tbl, union, eLine, fLine):
		fm, el = len(fLine), len(eLine)
		for j in xrange(fm):
			if sum([1 if tbl[i][j] else 0 for i in xrange(el)]) == 0: # found a hole
				for i in xrange(el):
					tbl[i][j] = union[i][j] 
		return tbl

	# extract phrases
	def extractPhrases(self, tbl, eLine, fLine):
		fm, el = len(fLine), len(eLine)
		efgrams = []
		# extract fNgram -> eUnigram
		for i, e in enumerate(eLine):
			fphrase = [f for j, f in enumerate(fLine) if tbl[i][j]]
			if len(fphrase) > 1 and True in tbl[i]:
				j = tbl[i].index(True)
				efgrams.append(([e], fphrase, i, j))
		# extract fUnigram -> eNgram
		for j, f in enumerate(fLine):
			ephrase = [e for i, e in enumerate(eLine) if tbl[i][j]]
			if len(ephrase) == 1 and f in self.LMT.dict1:
				self.phraseTable[f] = self.LMT.dict1[f] # map fUnigram -> eUnigram by using IBM model 1
			lst = [tbl[i][j] for i in xrange(el)]
			if True in lst:
				i = lst.index(True)
				efgrams.append((ephrase, [f], i, j))
		# sort efgrams ascendingly by i value
		efgrams.sort(key = lambda x: x[2])
		# fix simple reverse
		efgrams += self.fixReverse(efgrams)
		efgrams.sort(key = lambda x: x[2])
		# combine phrases
		combined = self.combinePhrases([], efgrams)
		# add these phrases to the phrase table
		for tpl in efgrams + combined:
			ephrase = ' '.join(self.removeConsecutiveDuplicate(tpl[0]))
			fphrase = ' '.join(self.removeConsecutiveDuplicate(tpl[1]))
			if fphrase not in self.phraseTable:
				self.phraseTable[fphrase] = ephrase # ptable[fphrase] = ephrase

	# remove consecutive duplicate
	def removeConsecutiveDuplicate(self, phrase):
		lst = []
		for word in phrase:
			if not lst or word != lst[-1]:
				lst.append(word)
		return lst

	# fix simple reverse
	def fixReverse(self, efgrams):
		efgramsPlus = []
		for i in xrange(1, len(efgrams)):
			tpl, first = efgrams[i-1], efgrams[i]
			if tpl[2] + len(tpl[0]) == first[2] and tpl[3] - len(first[1]) == first[3]:
				efgramsPlus.append((tpl[0] + first[0], tpl[1] + first[1], tpl[2], first[3]))
		return efgramsPlus

	# combine phrases
	def combinePhrases(self, partial, remaining):
		if not remaining:
			return partial
		first, rest = remaining[0], remaining[1:]
		combined = [first]
		for tpl in partial:
			if tpl[2] + len(tpl[0]) == first[2] and tpl[3] + len(tpl[1]) == first[3]:
				newtpl = (tpl[0] + first[0], tpl[1] + first[1], tpl[2], tpl[3])
				combined.append(newtpl)
		combined = self.combinePhrases(combined, rest)
		return partial + combined

	# compute the probability of the phrases
	def computePhraseProbabilities(self):
		total, count = 0, defaultdict(lambda: 0)
		for (eLine, fLine) in self.corpus:
			eLine = [e for e in eLine if e != 'NULL']
			fLine = [f for f in fLine if f != 'NULL']
			eLine, fLine = ' '.join(eLine), ' '.join(fLine)
			for fphrase in self.phraseTable:
				ephrase = self.phraseTable[fphrase]
				if ephrase in eLine and fphrase in fLine:
					count[(ephrase, fphrase)] += 1
					total += 1
		for fphrase in self.phraseTable:
			ephrase = self.phraseTable[fphrase]
			tpl = (ephrase, fphrase)
			self.phraseProbs[tpl] = float(count[tpl]) / float(total)

	# get phrase probability
	def getProbability(self, ephrase, fphrase):
		tpl = (ephrase, fphrase)
		return self.phraseProbs[tpl] if tpl in self.phraseProbs else 0
		
	# get ephrase
	def getPhrase(self, fphrase):
		return self.phraseTable[fphrase] if fphrase in self.phraseTable else None


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

	if len(sys.argv) > 7: # t2[f][e]
		print 'Reading t2 matrix...'
		t2 = IBMModel1.readWholeT(sys.argv[7])
	else:
		print 'Training IBM Model 1 t2...'
		t2 = IBMModel1.train(corpus, nIt, reverseCorpus = False, reverseT = True, alt = True) # reverse has higher accuracy

	print 'Creating language model...'
	eLM = LanguageModel.TrigramLanguageModel('e')
	eLM.train(corpus)
	LMT = LanguageModel.LMTranslator(t1, corpus, eLM, build = True)

	print 'Creating phrase table...'
	PT = PhraseTable(t1, t2, corpus, eLM, LMT, build = True)

	print 'Translating...'
	fTest = open(sys.argv[4], 'r')
	output = open(sys.argv[5], 'w+')
	for line in fTest:
		eLine = PT.translateSentence(line)
		output.write(eLine if eLine.endswith('\n') else eLine + '\n')
	fTest.close()
	output.close()


if __name__ == "__main__":
	main()
