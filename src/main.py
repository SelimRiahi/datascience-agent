# main.py - TON POINT DE DÉPART ! 🚀
"""
Agent Data Scientist Autonome - Version Débutant
Commencer par exécuter ce fichier : python main.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# L'agent principal (version simplifiée pour débuter)
class SimpleAutonomousDataScientist:
    """Version simplifiée de l'agent pour débuter"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {}
        print("🤖 Agent Data Scientist Autonome initialisé !")
        print("=" * 50)
    
    def load_data(self, file_path):
        """Charger les données automatiquement"""
        print(f"📊 Chargement des données depuis {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                print("❌ Format non supporté. Utilise .csv ou .xlsx")
                return False
            
            print(f"✅ Données chargées : {self.data.shape[0]} lignes, {self.data.shape[1]} colonnes")
            print(f"📋 Colonnes : {list(self.data.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return False
    
    def analyze_data_quality(self):
        """Analyser automatiquement la qualité des données"""
        print("\n🔍 ANALYSE AUTOMATIQUE DE LA QUALITÉ")
        print("-" * 40)
        
        if self.data is None:
            print("❌ Aucune donnée chargée !")
            return
        
        # Informations générales
        print(f"📊 Forme des données : {self.data.shape}")
        print(f"💾 Mémoire utilisée : {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Valeurs manquantes
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Valeurs manquantes détectées :")
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.data)) * 100
                print(f"   • {col}: {count} ({percentage:.1f}%)")
        else:
            print("✅ Aucune valeur manquante !")
        
        # Types de données
        print(f"\n📈 Types de données :")
        for dtype, columns in self.data.dtypes.groupby(self.data.dtypes).items():
            print(f"   • {dtype}: {len(columns)} colonnes")
        
        # Doublons
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            print(f"\n🔄 Doublons : {duplicates} lignes")
        else:
            print("✅ Aucun doublon détecté")
        
        # Statistiques descriptives
        print(f"\n📊 Aperçu des données :")
        print(self.data.head())
    
    def auto_clean_data(self):
        """Nettoyer automatiquement les données"""
        print("\n🧹 NETTOYAGE AUTOMATIQUE DES DONNÉES")
        print("-" * 40)
        
        if self.data is None:
            print("❌ Aucune donnée à nettoyer !")
            return
        
        original_shape = self.data.shape
        
        # Supprimer les doublons
        before_dup = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_dup = before_dup - len(self.data)
        if removed_dup > 0:
            print(f"✅ {removed_dup} doublons supprimés")
        
        # Gérer les valeurs manquantes
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype == 'object':
                    # Variables catégorielles : mode
                    mode_value = self.data[column].mode().iloc[0] if len(self.data[column].mode()) > 0 else 'Unknown'
                    self.data[column].fillna(mode_value, inplace=True)
                    print(f"✅ {column}: valeurs manquantes → '{mode_value}'")
                else:
                    # Variables numériques : médiane
                    median_value = self.data[column].median()
                    self.data[column].fillna(median_value, inplace=True)
                    print(f"✅ {column}: valeurs manquantes → {median_value:.2f}")
        
        print(f"📊 Forme finale : {original_shape} → {self.data.shape}")
    
    def auto_model_selection(self, target_column):
        """Sélectionner et entraîner automatiquement le meilleur modèle"""
        print(f"\n🤖 SÉLECTION AUTOMATIQUE DU MODÈLE")
        print("-" * 40)
        
        if self.data is None:
            print("❌ Aucune donnée disponible !")
            return
        
        if target_column not in self.data.columns:
            print(f"❌ Colonne '{target_column}' introuvable !")
            print(f"Colonnes disponibles : {list(self.data.columns)}")
            return
        
        # Préparer les données
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        # Séparer X et y
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Encoder les variables catégorielles
        encoders = {}
        for column in X.columns:
            if X[column].dtype == 'object':
                encoders[column] = LabelEncoder()
                X[column] = encoders[column].fit_transform(X[column].astype(str))
        
        # Détecter le type de problème
        if y.dtype == 'object' or y.nunique() < 10:
            problem_type = 'classification'
            if y.dtype == 'object':
                y_encoder = LabelEncoder()
                y = y_encoder.fit_transform(y)
        else:
            problem_type = 'regression'
        
        print(f"🎯 Type de problème détecté : {problem_type}")
        print(f"📊 Features : {X.shape[1]}")
        print(f"📈 Échantillons : {X.shape[0]}")
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Sélection et entraînement du modèle
        if problem_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric_name = "Précision"
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            metric_name = "RMSE"
        
        print(f"🏆 Modèle sélectionné : Random Forest")
        print(f"🎯 {metric_name} : {score:.4f}")
        
        # Importance des features
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔑 Top 5 features importantes :")
            for idx, row in feature_importance.head().iterrows():
                print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        # Sauvegarder les résultats
        self.results = {
            'model_type': 'Random Forest',
            'problem_type': problem_type,
            'score': score,
            'metric': metric_name,
            'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None,
            'encoders': encoders
        }
        
        return self.results
    
    def generate_insights(self):
        """Générer des insights automatiquement"""
        print(f"\n💡 INSIGHTS AUTOMATIQUES")
        print("-" * 40)
        
        if not self.results:
            print("❌ Aucun modèle entraîné !")
            return
        
        insights = []
        
        # Analyse de performance
        score = self.results['score']
        if self.results['problem_type'] == 'classification':
            if score > 0.9:
                insights.append(f"🎯 Excellente précision ({score:.1%}) - Modèle prêt pour production")
            elif score > 0.8:
                insights.append(f"📊 Bonne précision ({score:.1%}) - Modèle fiable")
            else:
                insights.append(f"⚠️ Précision modérée ({score:.1%}) - À améliorer")
        
        # Top features
        if self.results['feature_importance'] is not None:
            top_feature = self.results['feature_importance'].iloc[0]
            insights.append(f"🔑 Feature la plus importante : {top_feature['feature']} ({top_feature['importance']:.1%})")
        
        # Recommandations
        insights.append("✅ Modèle Random Forest sélectionné (robuste et interprétable)")
        insights.append("📊 Prêt pour déploiement ou analyse plus poussée")
        
        print("🔍 Insights découverts :")
        for insight in insights:
            print(f"   {insight}")
        
        return insights
    
    def save_model(self, filename="model.pkl"):
        """Sauvegarder le modèle"""
        if self.model is None:
            print("❌ Aucun modèle à sauvegarder !")
            return
        
        import pickle
        
        # Créer le dossier models s'il n'existe pas
        import os
        os.makedirs('models', exist_ok=True)
        
        filepath = f"models/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'results': self.results}, f)
        
        print(f"💾 Modèle sauvegardé : {filepath}")
    
    def create_simple_visualization(self):
        """Créer des visualisations simples"""
        print(f"\n📈 CRÉATION DE VISUALISATIONS")
        print("-" * 40)
        
        if self.data is None:
            print("❌ Aucune donnée pour visualiser !")
            return
        
        import matplotlib.pyplot as plt
        
        # Créer le dossier visualizations
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # Graphique 1: Distribution des variables numériques
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_columns[:4]):
                if i < len(axes):
                    axes[i].hist(self.data[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Fréquence')
            
            plt.tight_layout()
            plt.savefig('visualizations/distributions.png', dpi=300, bbox_inches='tight')
            print("✅ Graphique des distributions sauvegardé")
            plt.show()
        
        # Graphique 2: Feature Importance
        if self.results and self.results['feature_importance'] is not None:
            plt.figure(figsize=(10, 6))
            top_features = self.results['feature_importance'].head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Features Importantes')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            print("✅ Graphique d'importance des features sauvegardé")
            plt.show()

def main():
    """Fonction principale - COMMENCE ICI !"""
    print("🚀 DÉMARRAGE DE L'AGENT DATA SCIENTIST AUTONOME")
    print("=" * 60)
    
    # Créer l'agent
    agent = SimpleAutonomousDataScientist()
    
    # ÉTAPE 1: Charger tes données
    print("\n📁 ÉTAPE 1: Chargement des données")
    print("Télécharge le dataset Telco Customer Churn depuis Kaggle")
    print("Place le fichier dans le dossier 'data/'")
    
    # Remplace par le chemin vers ton fichier
    data_path = "C:\\Users\\Selim\\OneDrive\\Bureau\\data science agent\\data\\data.xlsx"
    
    # Vérifier si le fichier existe
    import os
    if not os.path.exists(data_path):
        print(f"❌ Fichier non trouvé : {data_path}")
        print("📥 Création de données d'exemple pour démonstration...")
        
        # Créer des données d'exemple
        os.makedirs('data', exist_ok=True)
        example_data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'age': np.random.normal(40, 12, 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'months_subscribed': np.random.randint(1, 60, 1000),
            'support_calls': np.random.poisson(2, 1000),
            'contract_type': np.random.choice(['Monthly', 'Yearly'], 1000),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic'], 1000),
            'churn': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7])
        })
        
        example_data.to_csv(data_path, index=False)
        print(f"✅ Données d'exemple créées : {data_path}")
    
    # Charger les données
    if agent.load_data(data_path):
        
        # ÉTAPE 2: Analyser la qualité
        agent.analyze_data_quality()
        
        # ÉTAPE 3: Nettoyer automatiquement
        agent.auto_clean_data()
        
        # ÉTAPE 4: Entraîner le modèle automatiquement
        target_column = 'churn'  # Change selon ton dataset
        results = agent.auto_model_selection(target_column)
        
        if results:
            # ÉTAPE 5: Générer des insights
            insights = agent.generate_insights()
            
            # ÉTAPE 6: Sauvegarder
            agent.save_model()
            
            # ÉTAPE 7: Visualisations
            agent.create_simple_visualization()
            
            print("\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS !")
            print("=" * 60)
            print("📁 Fichiers créés :")
            print("   • models/model.pkl (modèle sauvegardé)")
            print("   • visualizations/ (graphiques)")
            
            print(f"\n📊 Résumé :")
            print(f"   • Modèle : {results['model_type']}")
            print(f"   • Performance : {results['score']:.4f}")
            print(f"   • Prêt pour utilisation !")

if __name__ == "__main__":
    main()