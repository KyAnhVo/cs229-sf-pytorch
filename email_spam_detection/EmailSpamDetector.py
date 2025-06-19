from typing import Tuple, List, Dict

from nltk.corpus import stopwords
import NB
import sys
import csv
import re

def main():
    emails, y = getEmailsAndClassification()
    dictionary = buildSpamDictionary(emails= emails, y= y)
    print(dictionary)

def getEmailsAndClassification() -> Tuple[List[str], List[str]]:
    assert len(sys.argv) == 2 and sys.argv[1].endswith('.csv'), 'Expect CSV file'
    y = []
    emails = []

    with open(file= sys.argv[1], mode= 'r', encoding= 'utf-8') as file:
        reader = csv.reader(file, delimiter= ',')
        read = False
        for row in reader:
            # Skip first line
            if not read:
                read = True
                continue
            y.append(row[1])
            emails.append(row[2])
    
    y = [(elem == 'spam') for elem in y]
    return emails, y

def tokenize(email: str) -> List[str]:
    email = email.lower()
    words = re.findall(r'\b[a-z]+\b', email)
    return words

def buildSpamDictionary(emails: List[str], y: List[bool]) -> Dict[str, int]:
    stops = set(stopwords.words('english'))
    dictionary = {}
    for i in range(len(emails)):
        if not y[i]:    # skip if not spam
            continue
        words = tokenize(emails[i])
        for word in words:
            if word in stops:
                continue
            dictionary[word] = dictionary.get(word, 0) + 1
    return dictionary

if __name__ == '__main__':
    main()
