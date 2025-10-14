# mvp_load_model.py - MVP para load_model con VectorOfThought y hipótesis de payloads
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
import os
import logging
import re
from typing import Dict, Callable, Optional, Any
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import requests
import google.generativeai as genai  # Para Gemini

logging.basicConfig(level=logging.INFO)

# Tus clases del código (PatternStorage, GeminiClient, VectorOfThought - adaptadas pa' MVP)
class PatternStorage:
    def __init__(self, patterns_file='patterns.npy', vectors_file='vectors.npy'):
        self.patterns_file = patterns_file
        self.vectors_file = vectors_file

    def save_patterns(self, patterns, vectors):
        np.save(self.patterns_file, patterns)
        np.save(self.vectors_file, vectors)

    def load_patterns(self):
        try:
            patterns = np.load(self.patterns_file, allow_pickle=True).tolist()
            vectors = np.load(self.vectors_file, allow_pickle=True).tolist()
            return patterns, vectors
        except FileNotFoundError:
            return [], []

class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("La clave de la API de Gemini no fue proporcionada.")
        self.api_key = api_key
        genai.configure(api_key=api_key)  # Config global pa' genai
        self.model = "models/embedding-001"  # Embedding model

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            result = genai.embed_content(model=self.model, content=text)
            embedding = result['embedding']
            return np.array(embedding)
        except Exception as e:
            logging.error(f"Error al llamar a Gemini: {e}")
            return None

class VectorOfThought:
    HEURISTICS: Dict[str, Callable[[str, int], int]] = {
        "log2": lambda text, count: int(np.log2(count + 1) * 1000) % 1000 if count > 0 else 0,
        "dfs": lambda text, count: sum(ord(c) for c in text) % 1024,
        "custom_quadratic": lambda text, count: (sum([ord(c)**2 for c in text]) % 2048)
    }

    def __init__(self, n_clusters: int = 10, gemini_api_key: str = ""):
        logging.info("Inicializando VectorOfThought...")
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.storage = PatternStorage()
        self.gemini_client = GeminiClient(api_key=gemini_api_key) if gemini_api_key else None
        self.patterns, self.vectors = self.storage.load_patterns()
        self._update_clustering()

    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        if self.gemini_client:
            vector = self.gemini_client.get_embedding(text)
            return normalize(vector.reshape(1, -1))[0] if vector is not None else None
        else:
            # Fallback dummy pa' tests
            return normalize(np.random.rand(1, 768))[0]

    def _update_clustering(self):
        if len(self.vectors) >= self.n_clusters:
            vectors_stack = np.vstack(self.vectors)
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_model.fit(vectors_stack)
            logging.info("Clustering actualizado.")

    def add_pattern(self, text: str, heuristic_algorithm: str = "log2"):
        if heuristic_algorithm not in self.HEURISTICS:
            raise ValueError(f"Algoritmo '{heuristic_algorithm}' no reconocido.")
        
        new_vector = self._text_to_vector(text)
        if new_vector is None:
            logging.error("No vector generado.")
            return None, None
        
        existing_idx = next((i for i, p in enumerate(self.patterns) if p['text'] == text), None)
        
        if existing_idx is not None:
            old_vector = self.vectors[existing_idx]
            dot_product = np.dot(old_vector, new_vector)
            change_rate = 1 - dot_product
            self.vectors[existing_idx] = new_vector
            idx = self.patterns[existing_idx]['index']
        else:
            heuristic_func = self.HEURISTICS[heuristic_algorithm]
            idx = heuristic_func(text, len(self.patterns))
            self.patterns.append({"index": idx, "text": text, "concepto": "payload"})  # Tokenizado como concepto
            self.vectors.append(new_vector)
            change_rate = 0.0
        
        self._update_clustering()
        self.storage.save_patterns(self.patterns, self.vectors)
        return idx, change_rate

    def find_similar_pattern(self, text: str, search_metric: str = 'cosine') -> Optional[Dict]:
        if not self.vectors:
            return None

        query_vector = self._text_to_vector(text)
        if query_vector is None:
            return None
        query_vector = query_vector.reshape(1, -1)
        
        if self.cluster_model:
            cluster_idx = self.cluster_model.predict(query_vector)[0]
            member_indices = [i for i, label in enumerate(self.cluster_model.labels_) if label == cluster_idx]
            if not member_indices:
                return None
            vectors_in_cluster = np.vstack([self.vectors[i] for i in member_indices])
            distances = cdist(query_vector, vectors_in_cluster, metric=search_metric)[0]
            nearest_local_idx = np.argmin(distances)
            nearest_global_idx = member_indices[nearest_local_idx]
        else:
            vectors_stack = np.vstack(self.vectors)
            distances = cdist(query_vector, vectors_stack, metric=search_metric)[0]
            nearest_global_idx = np.argmin(distances)
        
        similarity_score = 1 - distances[nearest_local_idx] if search_metric == 'cosine' else distances[nearest_local_idx]
        similar_pattern = self.patterns[nearest_global_idx]
        logging.info(f"Similaridad: {similarity_score:.4f} para '{similar_pattern['text']}'")
        return similar_pattern

def _calculate_entropy(text: str) -> float:
    if len(text) == 0:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def load_model(path_to_pkl: str, compute_cm: bool = True, vector_of_relation: bool = True, entropy_baseline: float = 4.0, gemini_api_key: str = "") -> Dict[str, Any]:
    """
    MVP: Carga .pkl, producer básico, normaliza, CM + VOT pa' hipótesis de payloads.
    Uso: result = load_model('output/security_model.pkl', gemini_api_key=os.getenv('GEMINI_API_KEY'))
    Para detectar: similar = result['vector_of_thought'].find_similar_pattern(new_payload)
    """
    logging.info(f"🚀 Cargando modelo de {path_to_pkl}")
    
    # Carga .pkl
    with open(path_to_pkl, 'rb') as f:
        pkg = pickle.load(f)
    model: RandomForestClassifier = pkg['model']
    vectorizer: TfidfVectorizer = pkg['vectorizer']
    
    # Carga training_set
    training_csv = path_to_pkl.replace('.pkl', '.csv')
    if os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        texts = df['text'].tolist()
        y_true = df['label'].tolist()
        logging.info(f"Training_set cargado: {len(texts)} muestras")
    else:
        raise FileNotFoundError(f"No se encontró {training_csv}. Genera con el pipeline.")
    
    # Producer básico: Features + predicciones
    X_tfidf = vectorizer.transform(texts)
    def extract_features(texts):  # Heurísticas como en pipeline
        features = []
        for text in texts:
            feat = {
                'length': len(text),
                'special_chars': len(re.findall(r'[<>;=\'\"&|%]', text)),
                'sql_keywords': len(re.findall(r'\b(SELECT|UNION|DROP|INSERT|UPDATE|DELETE|EXEC)\b', text, re.IGNORECASE)),
                'xss_patterns': len(re.findall(r'<script|javascript:|on\w+=', text, re.IGNORECASE)),
                'path_traversal': len(re.findall(r'\.\./|\.\.\\|etc/passwd|win\.ini', text, re.IGNORECASE)),
                'entropy': _calculate_entropy(text),
                'url_encoded': len(re.findall(r'%[0-9a-fA-F]{2}', text)),
                'whitespace_ratio': len(re.findall(r'\s', text)) / max(1, len(text))
            }
            features.append(list(feat.values()))
        return np.array(features)
    
    X_advanced = extract_features(texts)
    X_combined = np.hstack([X_tfidf.toarray(), X_advanced])
    
    # Normaliza salida con training input
    X_normalized = normalize(X_combined, norm='l2')
    y_pred = model.predict(X_normalized)
    logging.info(f"Predicciones producidas: {sum(y_pred)} ataques detectados")
    
    # Matriz de confusión
    cm = None
    if compute_cm:
        cm = confusion_matrix(y_true, y_pred)
        print("Matriz de Confusión:\n", cm)
        logging.info(f"CM generada: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # VectorOfThought pa' relación/discriminación (hipótesis)
    vot = None
    relation_vector = None
    if vector_of_relation:
        vot = VectorOfThought(n_clusters=5, gemini_api_key=gemini_api_key)
        # Tokeniza y agrega payloads como conceptos (muestra pa' MVP)
        for i, text in enumerate(texts[:20]):  # Escala a más en prod
            concepto = "payload"  # Tokeniza: e.g., si tiene ' OR ', "SQLi concepto"
            if "' OR '" in text:
                concepto = "SQLi_payload"
            vot.add_pattern(text, "log2")
        
        # Ejemplo detección en tránsito
        test_payload = "/login?user=admin' OR '1'='1' -- path tokenizado"
        similar = vot.find_similar_pattern(test_payload)
        if similar:
            relation_vector = similar
            logging.info(f"Vector de relación: '{similar['text']}' (concepto: {similar.get('concepto', 'payload')})")
        
        # Chequeo entropía baseline
        test_entropy = _calculate_entropy(test_payload)
        if test_entropy > entropy_baseline:
            logging.info(f"🚨 Alta entropía ({test_entropy:.2f} > {entropy_baseline}): Ataque probable")
        else:
            logging.info(f"Entropía OK: {test_entropy:.2f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'y_pred': y_pred,
        'X_normalized': X_normalized,
        'confusion_matrix': cm,
        'vector_of_thought': vot,
        'relation_vector': relation_vector,  # Dict con similar payload/concepto
        'entropy_baseline': entropy_baseline
    }

# Ejemplo de uso (pon tu key)
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # O usa kaggle_secrets
    result = load_model("output/security_model.pkl", gemini_api_key=GEMINI_API_KEY)
    
    # Prueba detección
    if result['vector_of_thought']:
        new_transit = "<script>alert(1)</script> en /search"
        similar = result['vector_of_thought'].find_similar_pattern(new_transit)
        print(f"Detección en tránsito: Similar a '{similar['text']}' si existe.")
    
    print("MVP listo! Integra con Kafka pa' producer real.")
