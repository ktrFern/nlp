import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

FILENAME = "text.txt"

def has_resource(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False

if not (has_resource("tokenizers/punkt_tab") or has_resource("tokenizers/punkt")):
    for name in ("punkt_tab", "punkt"):
        nltk.download(name, quiet=True)

morph = pymorphy3.MorphAnalyzer()
POS_OK = {"NOUN", "ADJF"}
CASE_FIX = {"gen2": "gent", "acc2": "accs", "loc2": "loct"}

def norm_case(c):
    return CASE_FIX.get(c, c)

def agree(p1, p2):
    t1, t2 = p1.tag, p2.tag
    if t1.number is None or t1.number != t2.number:
        return False
    if t1.case is None or norm_case(t1.case) != norm_case(t2.case):
        return False
    both_nouns = t1.POS == "NOUN" and t2.POS == "NOUN"
    if (t1.number == "sing" or both_nouns) and t1.gender != t2.gender:
        return False
    return True

def good_parses(word):
    return [p for p in morph.parse(word) if p.tag.POS in POS_OK and "Apro" not in p.tag and "Anum" not in p.tag and "Abbr" not in p.tag]

def find_pairs(text):
    result = []
    for sent in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(sent, language="russian")
        for w1, w2 in zip(tokens, tokens[1:]):
            if not (w1.isalpha() and w2.isalpha()):
                continue
            best, best_score = None, -1
            for p1 in good_parses(w1):
                for p2 in good_parses(w2):
                    if agree(p1, p2) and p1.score * p2.score > best_score:
                        best_score = p1.score * p2.score
                        best = (p1, p2)
            if best:
                p1, p2 = best
                result.append({"lemmas": f"{p1.normal_form} {p2.normal_form}", "forms": f"{w1.lower()} {w2.lower()}", "analysis": f"{p1.tag} + {p2.tag}"})
    return result

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FILENAME)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pairs = find_pairs(text)
    print(f"Количество пар: {len(pairs)}\n")
    for i, p in enumerate(pairs, 1):
        print(f"{i}. {p['lemmas']} ({p['forms']}): {p['analysis']}")