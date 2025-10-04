# matrix_mechanic.py
"""
MECHANIC: Analiza .pkl + .npy → Matriz de Confusión + Relaciones Editables
Cielo entregado: Sí, pero con código terrenal 🛠️
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Para servidor sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

class MatrixMechanic:
    def __init__(self, model_path: str, dataset_path: str, patterns_path: str = None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.patterns_path = patterns_path
        self.model = None
        self.vectorizer = None
        self.dataset = None
        self.relations = {}
        self.confusion_data = {}
        
        # Cargar todo al iniciar
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """Cargar modelo, dataset y patrones existentes"""
        print("🔧 MECHANIC: Cargando componentes...")
        
        # 1. Cargar modelo .pkl
        with open(self.model_path, 'rb') as f:
            pkg = pickle.load(f)
        self.model = pkg['model']
        self.vectorizer = pkg['vectorizer']
        print(f"✅ Modelo cargado: {type(self.model).__name__}")
        
        # 2. Cargar dataset de entrenamiento
        self.dataset = pd.read_csv(self.dataset_path)
        print(f"✅ Dataset cargado: {len(self.dataset)} muestras")
        
        # 3. Cargar patrones .npy si existen
        if self.patterns_path and os.path.exists(self.patterns_path):
            try:
                patterns_data = np.load(self.patterns_path, allow_pickle=True).item()
                self.relations = patterns_data.get('relations', {})
                print(f"✅ Relaciones cargadas: {len(self.relations)} patrones")
            except:
                print("⚠️ No se pudieron cargar relaciones existentes")
        
    def generate_confusion_matrix(self, test_size: float = 0.3):
        """Generar matriz de confusión detallada"""
        from sklearn.model_selection import train_test_split
        
        print("🎯 Generando matriz de confusión...")
        
        # Preparar datos
        X = self.vectorizer.transform(self.dataset['text'])
        y = self.dataset['label']
        
        # Split train/test (usando mismo random_state para consistencia)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Predecir y calcular matriz
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Estadísticas detalladas
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Guardar datos de confusión
        self.confusion_data = {
            'matrix': cm.tolist(),
            'report': report,
            'test_size': len(X_test),
            'true_positives': int(cm[1, 1]),
            'false_positives': int(cm[0, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_negatives': int(cm[1, 0]),
            'accuracy': report['accuracy'],
            'precision_malicious': report['1']['precision'],
            'recall_malicious': report['1']['recall'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Matriz generada - Accuracy: {report['accuracy']:.3f}")
        return self.confusion_data
    
    def visualize_confusion(self, save_path: str = "output/confusion_matrix.png"):
        """Visualizar matriz de confusión"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = np.array(self.confusion_data['matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Ataque'],
                   yticklabels=['Normal', 'Ataque'])
        plt.title('Matriz de Confusión - Detección de Amenazas')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Añadir métricas en el gráfico
        accuracy = self.confusion_data['accuracy']
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matriz guardada en: {save_path}")
        return save_path
    
    def analyze_relationship_errors(self):
        """Analizar qué relaciones/predicciones fallaron"""
        from sklearn.model_selection import train_test_split
        
        X = self.vectorizer.transform(self.dataset['text'])
        y = self.dataset['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        y_pred = self.model.predict(X_test)
        
        # Identificar errores
        errors = {
            'false_positives': [],  # Normales clasificados como ataques
            'false_negatives': [],  # Ataques clasificados como normales
            'correct_predictions': []  # Para referencia
        }
        
        test_texts = self.dataset.iloc[X_test.index]['text'].values
        sources = self.dataset.iloc[X_test.index]['source'].values
        
        for i, (true, pred, text, source) in enumerate(zip(y_test, y_pred, test_texts, sources)):
            if true == 0 and pred == 1:  # Falso positivo
                errors['false_positives'].append({
                    'text': text,
                    'source': source,
                    'confidence': max(self.model.predict_proba(X_test[i])[0]),
                    'index': int(X_test.index[i])
                })
            elif true == 1 and pred == 0:  # Falso negativo
                errors['false_negatives'].append({
                    'text': text,
                    'source': source, 
                    'confidence': max(self.model.predict_proba(X_test[i])[0]),
                    'index': int(X_test.index[i])
                })
            else:  # Correcto
                errors['correct_predictions'].append({
                    'text': text,
                    'source': source,
                    'true_label': true,
                    'pred_label': pred,
                    'confidence': max(self.model.predict_proba(X_test[i])[0])
                })
        
        print(f"🔍 Errores encontrados: {len(errors['false_positives'])} FP, {len(errors['false_negatives'])} FN")
        return errors
    
    def export_relations(self, output_path: str = "output/relations_export.json"):
        """Exportar relaciones con pesos editables"""
        # Generar relaciones basadas en el análisis de errores
        errors = self.analyze_relationship_errors()
        
        relations_export = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'model_accuracy': self.confusion_data['accuracy'],
                'total_patterns': len(self.relations),
                'false_positives': len(errors['false_positives']),
                'false_negatives': len(errors['false_negatives'])
            },
            'relations': self.relations,
            'error_analysis': errors,
            'confusion_matrix': self.confusion_data,
            'editable_weights': self._generate_editable_weights()
        }
        
        # Guardar como JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relations_export, f, indent=2, ensure_ascii=False)
        
        print(f"📦 Relaciones exportadas: {output_path}")
        return output_path
    
    def _generate_editable_weights(self):
        """Generar estructura de pesos editables manualmente"""
        editable_weights = {}
        
        # Patrones del dataset como base
        for source in self.dataset['source'].unique():
            source_data = self.dataset[self.dataset['source'] == source]
            editable_weights[source] = {
                'current_weight': 1.0,  # Peso actual (editable)
                'suggested_weight': self._suggest_weight(source),
                'samples_count': len(source_data),
                'accuracy_by_source': self._calculate_source_accuracy(source),
                'patterns': source_data['text'].tolist()[:10]  # Primeros 10 patrones
            }
        
        return editable_weights
    
    def _suggest_weight(self, source: str):
        """Sugerir peso basado en rendimiento"""
        source_data = self.dataset[self.dataset['source'] == source]
        accuracy = self._calculate_source_accuracy(source)
        
        if accuracy > 0.9:
            return 1.2  # Aumentar peso
        elif accuracy < 0.7:
            return 0.8  # Reducir peso
        else:
            return 1.0  # Mantener
    
    def _calculate_source_accuracy(self, source: str):
        """Calcular accuracy por fuente (simplificado)"""
        # En implementación real, usaría cross-validation
        source_data = self.dataset[self.dataset['source'] == source]
        if len(source_data) == 0:
            return 0.0
        
        # Simular accuracy basado en distribución de labels
        malicious_ratio = source_data['label'].mean()
        return 0.8 + (malicious_ratio * 0.2)  # Heurística simple
    
    def manual_weight_edit(self, source: str, new_weight: float):
        """Editar manualmente el peso de una fuente"""
        if source in self._generate_editable_weights():
            # Aquí integrarías con el modelo para re-entrenar con nuevos pesos
            print(f"✏️ Peso editado: {source} -> {new_weight}")
            
            # Actualizar relaciones
            if source not in self.relations:
                self.relations[source] = {}
            self.relations[source]['manual_weight'] = new_weight
            self.relations[source]['last_edited'] = datetime.now().isoformat()
            
            return True
        return False

# 🎯 EJECUCIÓN COMPLETA DEL MECHANIC
def run_mechanic_analysis():
    """Ejecutar análisis completo del Mechanic"""
    print("🛠️ INICIANDO MATRIX MECHANIC...")
    print("=" * 50)
    
    # Configurar paths (ajustar según tu estructura)
    model_path = "ThePipeLine/output/security_model.pkl"
    dataset_path = "ThePipeLine/output/training_dataset.csv" 
    patterns_path = "ThePipeLine/output/patterns.npy"  # Opcional
    
    # Inicializar Mechanic
    mechanic = MatrixMechanic(model_path, dataset_path, patterns_path)
    
    # 1. Generar matriz de confusión
    confusion_data = mechanic.generate_confusion_matrix()
    
    # 2. Visualizar
    viz_path = mechanic.visualize_confusion()
    
    # 3. Analizar errores
    errors = mechanic.analyze_relationship_errors()
    
    # 4. Exportar relaciones editables
    export_path = mechanic.export_relations()
    
    # 5. Mostrar resumen
    print("\n📈 RESUMEN MECHANIC:")
    print(f"   • Accuracy: {confusion_data['accuracy']:.3f}")
    print(f"   • Precisión (Ataques): {confusion_data['precision_malicious']:.3f}")
    print(f"   • Recall (Ataques): {confusion_data['recall_malicious']:.3f}")
    print(f"   • Falsos Positivos: {confusion_data['false_positives']}")
    print(f"   • Falsos Negativos: {confusion_data['false_negatives']}")
    print(f"   • Visualización: {viz_path}")
    print(f"   • Relaciones: {export_path}")
    
    return mechanic

if __name__ == "__main__":
    mechanic = run_mechanic_analysis()
    
    # Ejemplo de edición manual
    print("\n🎛️  EJEMPLO EDICIÓN MANUAL:")
    mechanic.manual_weight_edit("owasp", 1.1)
    mechanic.manual_weight_edit("normal", 0.9)
    
    print("✅ MECHANIC COMPLETADO - Relaciones listas para ajuste manual")
