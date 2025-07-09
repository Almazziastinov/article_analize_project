import pandas as pd
import spacy
import huspacy
from textstat import textstat
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class HungarianTextAnalyzer:
    def __init__(self):
        # Hungarian stop words (example list, can be expanded)
        huspacy.download()
        self.nlp = spacy.load("hu_core_news_lg")
        self.stop_words = {
            'a', 'az', 'és', 'is', 'még', 'vagy', 'de', 'hogy', 'ez', 'egy',
            'nem', 'saját', 'mint', 'én', 'te', 'ő', 'mi', 'ti', 'ők', 'ön',
            'önök', 'én', 'neki', 'nekem', 'neked', 'nektek', 'nekik', 'önnek',
            'önöknek', 'velem', 'veled', 'vele', 'velünk', 'veletek', 'velük',
            'önnel', 'önökkel', 'nálam', 'nála', 'nálunk', 'náltok', 'náluk',
            'önnél', 'önöknél', 'hozzám', 'hozzád', 'hozzá', 'hozzánk', 'hozzátok',
            'hozzájuk', 'önhöz', 'önökhöz', 'tőlem', 'tőled', 'tőle', 'tőlünk',
            'tőletek', 'tőlük', 'öntőle', 'önöktől', 'értem', 'érted', 'érte',
            'értünk', 'értetek', 'értük', 'önért', 'önökért', 'nekem', 'neked',
            'neki', 'nekünk', 'nektek', 'nekik', 'önnek', 'önöknek'
        }
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words=list(self.stop_words))

    def analyze_text(self, text):
        """Main text analysis function"""
        doc = self.nlp(text)

        # Basic metrics
        metrics = {
            'character_count': len(text),
            'word_count': len([token for token in doc if not token.is_punct]),
            'sentence_count': len(list(doc.sents)),
            'average_sentence_length': np.mean([len(sent) for sent in doc.sents]) if len(list(doc.sents)) > 0 else 0,
            'paragraph_count': len(text.split('\n\n')),
            'unique_words': len(set(token.text.lower() for token in doc if not token.is_punct and token.text.lower() not in self.stop_words)),
            'lexical_diversity': self._calculate_ttr(doc),
            'readability': textstat.flesch_reading_ease(text),
        }

        # POS features
        pos_tags = [token.pos_ for token in doc]
        pos_counts = Counter(pos_tags)
        metrics.update({
            'noun_ratio': pos_counts.get('NOUN', 0) / len(pos_tags) if len(pos_tags) > 0 else 0,
            'verb_ratio': pos_counts.get('VERB', 0) / len(pos_tags) if len(pos_tags) > 0 else 0,
            'adjective_ratio': pos_counts.get('ADJ', 0) / len(pos_tags) if len(pos_tags) > 0 else 0,
            'average_dependency_depth': self._calculate_dependency_depth(doc),
        })

        # Structural elements
        metrics.update({
            'question_count': sum(1 for sent in doc.sents if '?' in sent.text),
            'list_items': sum(1 for token in doc if token.text in ('•', '-', '—', '*') and token.is_space),
            'number_count': sum(1 for token in doc if token.like_num),
            'quote_count': text.count('"') // 2,
        })

        return metrics

    def _calculate_ttr(self, doc):
        """Calculate lexical diversity (Type-Token Ratio)"""
        words = [token.text.lower() for token in doc if not token.is_punct and token.text.lower() not in self.stop_words]
        if not words:
            return 0
        return len(set(words)) / len(words)

    def _calculate_dependency_depth(self, doc):
        """Calculate average dependency depth"""
        depths = []
        for sentence in doc.sents:
            for token in sentence:
                depth = 0
                while token.head != token:
                    depth += 1
                    token = token.head
                depths.append(depth)
        return np.mean(depths) if depths else 0

    def analyze_corpus(self, texts, labels=None):
        """Analyze texts and return DataFrame"""
        results = []
        for i, text in enumerate(texts):
            metrics = self.analyze_text(text)
            if labels is not None:
                metrics['label'] = labels[i]
            results.append(metrics)

        df = pd.DataFrame(results)

        # Keyword analysis
        if len(texts) > 1:
            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                feature_names = self.vectorizer.get_feature_names_out()
                dense = tfidf_matrix.todense()
                df['keywords'] = [', '.join([feature_names[i] for i in np.argsort(dense[i]).A1[-3:][::-1]])
                                        for i in range(len(texts))]
            except ValueError:
                df['keywords'] = "Not enough data"

        return df

    def compare_groups(self, df, group_column='label'):
        """Compare groups"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        group_statistics = df.groupby(group_column)[numeric_columns].mean().T
        return group_statistics
