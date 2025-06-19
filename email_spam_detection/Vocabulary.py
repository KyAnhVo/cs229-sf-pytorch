from typing import Tuple, List, Dict

from nltk.corpus import stopwords
import NB
import sys
import csv
import re

class SpamVocabulary:
    emails: List[(str, bool)]
    vocabulary: Dict[str, int]
    activeVocabulary: Dict[str, int]
    activeVocabularySize: int

    def __init__(self, emails: List[Tuple[str, int]], activeVocabularySize: int = 1000):
        self.emails = emails
        self.activeVocabularySize = activeVocabularySize
        self._fillVocabulary()

    
    def _tokenize(self, string: str) -> List[str]:
        lowered = string.lower()
        return re.findall('\b[a-z]+\b', lowered)
    
    def _fillVocabulary(self):
        self.vocabulary = {}
        for email, isSpam in self.emails:
            if not isSpam:
                continue
            words = self._tokenize(email)
            for word in words:
                self.spamVocabulary[word] = self.spamVocabulary.get(word, 0) + 1
    
    # TODO: Optimize this function (heapq for O(log n) maybe)?
    def _fillActiveVocabulary(self):
        assert(self.vocabulary is not None), 'vocabulary is not instantiated'
        vocabList = sorted(
            iterable= list(self.vocabulary.items()),
            key= lambda x: x[1])
        for i in range(self.activeVocabularySize):
            if i >= len(vocabList):
                break
            word, appearance = vocabList[i]
            self.activeVocabulary[word] = appearance
