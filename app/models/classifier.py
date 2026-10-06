"""
Clasificadores de texto: a partir de la descripcion del reporte sugieren el
tipo de incidente. Dos modelos intercambiables (CLASSIFIER_MODEL):

    naive_bayes : Naive Bayes multinomial con suavizado de Laplace, entrenado
                  al arrancar con TRAINING_DATA. Devuelve probabilidades.
    keywords    : modelo base por palabras clave (sirve para comparar).
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Protocol, Tuple

from .training_data import TRAINING_DATA

STOPWORDS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "le", "lo", "los", "me", "mi", "nos", "o",
    "para", "por", "que", "se", "sin", "su", "un", "una", "unos", "y", "ya", "hay", "estaba", "esta", "muy",
}


def tokenize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r"[a-z]+", text)
    # "stemming" muy simple: quitamos plurales para que robos == robo
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        if len(w) > 4 and w.endswith("es"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


class Classifier(Protocol):
    name: str

    def predict(self, text: str) -> List[Tuple[str, float]]:
        """Lista de (codigo, probabilidad) de mayor a menor."""


class NaiveBayesClassifier:
    name = "naive_bayes"

    def __init__(self, data=TRAINING_DATA, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_docs: Counter = Counter()
        self.word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_words: Counter = Counter()
        self.vocab: set = set()
        self.fit(data)

    def fit(self, data) -> None:
        for text, label in data:
            self.class_docs[label] += 1
            for w in tokenize(text):
                self.word_counts[label][w] += 1
                self.total_words[label] += 1
                self.vocab.add(w)

    def predict(self, text: str) -> List[Tuple[str, float]]:
        tokens = tokenize(text)
        n_docs = sum(self.class_docs.values())
        v = len(self.vocab)
        log_probs = {}
        for label in self.class_docs:
            lp = math.log(self.class_docs[label] / n_docs)
            for w in tokens:
                lp += math.log((self.word_counts[label][w] + self.alpha) / (self.total_words[label] + self.alpha * v))
            log_probs[label] = lp
        # softmax para convertir a probabilidades
        m = max(log_probs.values())
        exp = {k: math.exp(val - m) for k, val in log_probs.items()}
        total = sum(exp.values())
        return sorted(((k, val / total) for k, val in exp.items()), key=lambda x: -x[1])


class KeywordClassifier:
    name = "keywords"

    KEYWORDS = {
        "armed_robbery": ["arma", "pistola", "revolver", "cuchillo", "navaja", "atraco", "atracaron", "asalto", "fleteo"],
        "assault": ["golpe", "pelea", "rina", "agresion", "agredieron", "herido", "pegaron"],
        "home_burglary": ["casa", "apartamento", "vivienda", "puerta", "ventana", "forzaron", "metieron"],
        "vehicle_theft": ["carro", "moto", "motocicleta", "vehiculo", "bicicleta", "camioneta", "llanta"],
        "theft": ["raponazo", "celular", "billetera", "bolso", "cosquilleo", "hurto", "arrebataron"],
        "drug_dealing": ["droga", "olla", "jibaro", "marihuana", "bazuco", "vicio", "microtrafico"],
        "vandalism": ["grafiti", "rompieron", "danaron", "lampara", "vidrio", "quemaron", "destruyeron"],
        "suspicious_activity": ["sospechoso", "sospechosa", "rondando", "merodeando", "vigilando", "extrano"],
    }

    def predict(self, text: str) -> List[Tuple[str, float]]:
        tokens = tokenize(text)
        scores = {}
        for label, keywords in self.KEYWORDS.items():
            stems = tokenize(" ".join(keywords))
            # comparamos por los primeros 5 caracteres (robo/robaron, golpe/golpearon...)
            hits = sum(1 for t in tokens if any(t[:5] == s[:5] for s in stems))
            scores[label] = 1 + 3 * hits
        total = sum(scores.values())
        return sorted(((k, v / total) for k, v in scores.items()), key=lambda x: -x[1])


CLASSIFIERS = {"naive_bayes": NaiveBayesClassifier, "keywords": KeywordClassifier}
