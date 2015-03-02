# NLTKPOS.py
# CS124, Winter 2015, PA6: Machine Translation
# 
# Group members:
#   Albert Chu (achu8)
#   Brad Huang (brad0309)
#   Nick Moores (npmoores)
#
# This script uses the Natural Language Toolkit's Parts-of-Speech tagger to do post-processing
# on the English translations generated from Spanish.

import nltk

# [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('beautful', 'JJ'), ('gift', 'NN'), ('.', '.'), 
#  ('I', 'PRP'), ('want', 'VBP'), ('to', 'TO'), ('see', 'VB'), ('this', 'DT'), ('beautiful', 'JJ'),
#  ('place', 'NN'), ('.', '.')]
# [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('Thursday', 'NNP'), ('morning', 'NN')]

def matchAny(token, tokens):
	for t in tokens:
		if token == t:
			return True
	return False

class NLTK_POS:
	def tokenize(self, sentence):
		tokens = nltk.word_tokenize(sentence)
		return tokens

	def tag(self, tokens):
		try:
			tagged = nltk.pos_tag(tokens)
		except UnicodeDecodeError:
			tagged = None
		return tagged

	def postprocessing(self, orig_tokens, tagged):
		tokens = list(orig_tokens)
		tokens, changed = self.postprocessing_unigrams(tokens, tagged)
		tokens, changed = self.postprocessing_bigrams(tokens, tagged) or changed
		return tokens, changed

	def postprocessing_unigrams(self, tokens, tagged):
		return tokens, False

	def postprocessing_bigrams(self, tokens, tagged):
		tokens, changed = self.ensureAdjectiveNoun(tokens, tagged)
		return tokens, changed

	def ensureProperNoun(self, tokens, tagged):
		changed = False
		for i in xrange(len(tokens)):
			if self.isProperNoun(tokens[i]):
				changed = True
				tokens[i] = tokens[i].capitalize()
		return tokens, changed

	def ensureAdjectiveNoun(self, tokens, tagged):
		changed = False
		for i in xrange(1, len(tokens)):
			if self.isAdjective(tagged[i][1]) and self.isNoun(tagged[i-1][1]):
				changed = True
				self.adjectiveNounSwaps.append(' '.join([tokens[i-1], tokens[i], '->', tokens[i], tokens[i-1]]))
				tokens[i-1], tokens[i] = tokens[i], tokens[i-1]
				tagged[i-1], tagged[i] = tagged[i], tagged[i-1]
		return tokens, changed

	def isNoun(self, token):
		nouns = [self.NounSingular, self.NounPlural]
		return matchAny(token, nouns) or self.isProperNoun(token)

	def isProperNoun(self, token):
		nouns = [self.ProperNounSingular, self.ProperNounPlural]
		return matchAny(token, nouns)

	def isAdjective(self, token):
		adjectives = [self.Adjective, token == self.AdjectiveComparative, token == self.AdjectiveSuperlative]
		return matchAny(token, adjectives)

	def isAdverb(self, token):
		adverbs = [self.Adverb, self.AdverbComparative, self.AdverbSuperlative]
		return matchAny(token, adverbs)

	def isVerb(self, token):
		verbs = [self.VerbBaseForm, self.VerbPastTense, self.VerbPresentParticiple, self.VerbPastParticiple,
				self.VerbNon3rdPersonSingularPresent, self.Verb3rdPersonSingularPresent]
		return matchAny(token, verbs)
		
	def isPronoun(self, token):
		pronouns = [self.PersonalPronoun, self.PossessivePronoun]
		return matchAny(token, pronouns)


	def __init__(self):
		self.adjectiveNounSwaps = []
		self.CoordinatingConjunction = 'CC'
		self.CardinalNumber = 'CD'
		self.Determiner = 'DT'
		self.ExistentialThere = 'EX'
		self.ForeignWord = 'FW'
		self.Preposition = 'IN'
		self.Adjective = 'JJ'
		self.AdjectiveComparative = 'JJR'
		self.AdjectiveSuperlative = 'JJS'
		self.ListItemMarker = 'LS'
		self.Modal = 'MD'
		self.NounSingular = 'NN'
		self.NounPlural = 'NNS'
		self.ProperNounSingular = 'NNP'
		self.ProperNounPlural = 'NNPS'
		self.Predeterminer = 'PDT'
		self.PossessiveEnding = 'POS'
		self.PersonalPronoun = 'PRP'
		self.PossessivePronoun = 'PRP$'
		self.Adverb = 'RB'
		self.AdverbComparative = 'RBR'
		self.AdverbSuperlative = 'RBS'
		self.Particle = 'RP'
		self.Symbol = 'SYM'
		self.To = 'TO'
		self.Interjection = 'UH'
		self.VerbBaseForm = 'VB'
		self.VerbPastTense = 'VBD'
		self.VerbPresentParticiple = 'VBG'
		self.VerbPastParticiple = 'VBN'
		self.VerbNon3rdPersonSingularPresent = 'VBP' # checks for we, you
		self.Verb3rdPersonSingularPresent = 'VBZ' # checks for he, she, they
		self.WhDeterminer = 'WDT'
		self.WhPronoun = 'WP'
		self.PossessiveWhPronoun = 'WP$'
		self.WhAdverb = 'WRB'

