import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
import json
from datetime import datetime
import warnings
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    BINARY = "binary"

class QualityIssue(Enum):
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMAT = "inconsistent_format"
    INVALID_VALUES = "invalid_values"

@dataclass
class ColumnProfile:
    name: str
    data_type: DataType
    business_context: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    duplicates: int
    outliers: List[int]
    quality_issues: List[QualityIssue]
    recommended_actions: List[str]
    confidence_score: float

class LlamaIntegration:
    """Interface pour l'intégration avec Azure Llama"""
    
    def __init__(self, model_path: str = None):
        try:
            import secrets_config
            self.client = ChatCompletionsClient(
                endpoint=secrets_config.AZURE_ENDPOINT,
                credential=AzureKeyCredential(secrets_config.AZURE_API_KEY),
                api_version="2024-05-01-preview"
            )
            self.model_name = secrets_config.MODEL_NAME
            print("Azure Llama connecté!")
        except Exception as e:
            raise Exception(f"Erreur connexion Azure: {e}")
        
    def analyze_business_context(self, column_name: str, sample_values: List[Any]) -> Tuple[str, float]:
        """Analyse le contexte métier d'une colonne"""
        prompt = f"""
        Analysez cette colonne de données :
        Nom: {column_name}
        Échantillon de valeurs: {sample_values[:10]}
        
        Déterminez le contexte métier de cette colonne (ex: identifiant client, montant financier, 
        catégorie produit, date de transaction, etc.) et votre niveau de confiance (0-1).
        
        Réponse format JSON: {{"context": "description", "confidence": 0.95}}
        """
        
        response = self.client.complete(
            messages=[
                SystemMessage(content="Vous êtes un expert en analyse de données. Répondez UNIQUEMENT en JSON valide."),
                UserMessage(content=prompt)
            ],
            max_tokens=300,
            temperature=0.1,
            model=self.model_name
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "{" in result_text and "}" in result_text:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            result_text = result_text[start:end]
        
        result = json.loads(result_text)
        return result["context"], result["confidence"]
    
    def get_cleaning_strategy(self, column_profile: ColumnProfile, sample_data: List[Any]) -> Dict[str, Any]:
        """LLM détermine la stratégie de nettoyage complète pour cette colonne"""
        prompt = f"""
        Vous êtes un expert data scientist. Analysez cette colonne et déterminez la stratégie de nettoyage appropriée.

        COLONNE: {column_profile.name}
        CONTEXTE MÉTIER: {column_profile.business_context}
        ÉCHANTILLON DE DONNÉES: {sample_data[:15]}
        PROBLÈMES DÉTECTÉS: {[issue.value for issue in column_profile.quality_issues]}
        VALEURS MANQUANTES: {column_profile.missing_percentage:.1f}%
        VALEURS UNIQUES: {column_profile.unique_count}

        RÈGLES DATA SCIENCE IMPORTANTES:
        1. Si c'est un identifiant/code: NE JAMAIS appliquer de statistiques (moyenne, outliers, etc.)
        2. Si quantité négative: souvent = retour/annulation → supprimer ligne
        3. Si identifiant commence par 'C': peut être annulation → supprimer ligne  
        4. Si prix ≤ 0: erreur de données → supprimer ligne
        5. Codes alphanumériques (ex: ABC123) sont NORMAUX pour identifiants
        6. Dates futures suspectes → valider
        7. Préserver l'intégrité métier avant tout

        Déterminez:
        - Le TYPE RÉEL de donnée (identifier/quantity/price/date/category/text)
        - Les ACTIONS spécifiques à prendre
        - Les VALEURS à supprimer/garder
        - Les RÈGLES business à appliquer

        Réponse JSON UNIQUEMENT:
        {{
            "data_category": "identifier|quantity|price|date|category|text",
            "cleaning_actions": ["action1", "action2"],
            "removal_rules": ["rule1", "rule2"],
            "preserve_data": true/false,
            "business_logic": "explication courte"
        }}
        """
        
        response = self.client.complete(
            messages=[
                SystemMessage(content="Expert data scientist. Répondez UNIQUEMENT en JSON valide."),
                UserMessage(content=prompt)
            ],
            max_tokens=600,
            temperature=0.1,
            model=self.model_name
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "{" in result_text and "}" in result_text:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            result_text = result_text[start:end]
        
        return json.loads(result_text)

class DataProfiler:
    """Analyse automatique de la qualité et profiling des données"""
    
    def __init__(self, llama_integration: LlamaIntegration):
        self.llama = llama_integration
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """Profile complet d'un dataset"""
        profiles = {}
        
        logger.info(f"Profiling dataset avec {len(df.columns)} colonnes et {len(df)} lignes")
        
        for col in df.columns:
            profiles[col] = self._profile_column(df, col)
        
        return profiles
    
    def _profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Profile une colonne spécifique"""
        col_data = df[column]
        
        # Détection automatique du type
        data_type = self._detect_data_type(col_data)
        
        # Analyse du contexte métier via Llama
        sample_values = col_data.dropna().head(10).tolist()
        business_context, confidence = self.llama.analyze_business_context(column, sample_values)
        
        # Calculs statistiques de base
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(col_data)) * 100
        unique_count = col_data.nunique()
        
        # Détection des problèmes de qualité
        quality_issues = self._detect_quality_issues(col_data, data_type)
        duplicates = len(col_data) - unique_count
        outliers = self._detect_outliers(col_data, data_type)
        
        profile = ColumnProfile(
            name=column,
            data_type=data_type,
            business_context=business_context,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            duplicates=duplicates,
            outliers=outliers,
            quality_issues=quality_issues,
            recommended_actions=[],
            confidence_score=confidence
        )
        
        # Génération des recommandations via Llama
        sample_values = col_data.dropna().head(20).tolist()
        cleaning_strategy = self.llama.get_cleaning_strategy(profile, sample_values)
        profile.recommended_actions = cleaning_strategy.get('cleaning_actions', [])
        
        return profile
    
    def _detect_data_type(self, series: pd.Series) -> DataType:
        """Détection intelligente du type de données"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return DataType.TEXT
        
        print(f"Analyse type pour: échantillon = {series_clean.head(5).tolist()}")
        
        # Test pour les valeurs binaires
        unique_vals = set(series_clean.astype(str).str.lower())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 'oui', 'non'}):
            print("Type détecté: BINARY")
            return DataType.BINARY
        
        # Test pour les valeurs numériques AVANT les dates
        if pd.api.types.is_numeric_dtype(series):
            print("Type détecté: NUMERIC (pandas dtype)")
            return DataType.NUMERIC
        
        # Test conversion numérique
        try:
            sample = series_clean.head(100)
            numeric_converted = pd.to_numeric(sample, errors='coerce')
            valid_numeric = numeric_converted.notna().sum()
            
            if valid_numeric / len(sample) > 0.8:
                print("Type détecté: NUMERIC (conversion)")
                return DataType.NUMERIC
        except:
            pass
        
        # Test pour les dates APRÈS les numériques
        try:
            sample = series_clean.head(50)
            date_converted = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
            valid_dates = date_converted.notna().sum()
            
            if valid_dates / len(sample) > 0.7:
                print("Type détecté: DATETIME")
                return DataType.DATETIME
        except:
            pass
        
        # Test pour les identifiants (unique values + format court)
        uniqueness_ratio = series.nunique() / len(series.dropna())
        avg_length = np.mean([len(str(x)) for x in series_clean.head(10)])
        
        if uniqueness_ratio > 0.9 and avg_length < 20:
            print("Type détecté: IDENTIFIER")
            return DataType.IDENTIFIER
        
        # Test pour les catégories (peu de valeurs uniques)
        if series.nunique() / len(series) < 0.1 and series.nunique() < 100:
            print("Type détecté: CATEGORICAL")
            return DataType.CATEGORICAL
        
        print("Type détecté: TEXT (fallback)")
        return DataType.TEXT
    
    def _detect_quality_issues(self, series: pd.Series, data_type: DataType) -> List[QualityIssue]:
        """Détection automatique des problèmes de qualité"""
        issues = []
        
        # Valeurs manquantes
        if series.isnull().sum() > 0:
            issues.append(QualityIssue.MISSING_VALUES)
        
        # Doublons (pour les identifiants)
        if data_type == DataType.IDENTIFIER and series.duplicated().sum() > 0:
            issues.append(QualityIssue.DUPLICATES)
        
        # Outliers pour données numériques
        if data_type == DataType.NUMERIC and len(self._detect_outliers(series, data_type)) > 0:
            issues.append(QualityIssue.OUTLIERS)
        
        # Formats inconsistants
        if self._has_inconsistent_format(series, data_type):
            issues.append(QualityIssue.INCONSISTENT_FORMAT)
        
        return issues
    
    def _detect_outliers(self, series: pd.Series, data_type: DataType) -> List[int]:
        """Détection d'outliers statistiques"""
        if data_type != DataType.NUMERIC:
            return []
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) < 4:
            return []
        
        # Méthode IQR
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        return numeric_series[outlier_mask].index.tolist()
    
    def _has_inconsistent_format(self, series: pd.Series, data_type: DataType) -> bool:
        """Détecte les formats inconsistants"""
        if data_type == DataType.DATETIME:
            # Vérifier si les dates ont des formats différents
            sample = series.dropna().head(20)
            formats_detected = set()
            
            for val in sample:
                val_str = str(val)
                if re.match(r'\d{4}-\d{2}-\d{2}', val_str):
                    formats_detected.add('YYYY-MM-DD')
                elif re.match(r'\d{2}/\d{2}/\d{4}', val_str):
                    formats_detected.add('MM/DD/YYYY')
                elif re.match(r'\d{2}-\d{2}-\d{4}', val_str):
                    formats_detected.add('MM-DD-YYYY')
            
            return len(formats_detected) > 1
        
        return False

class SmartCleaner:
    """Nettoyage entièrement piloté par LLM - zéro logique hardcodée"""
    
    def __init__(self, llama_integration: LlamaIntegration):
        self.llama = llama_integration
        self.cleaning_log = []
    
    def clean_dataset(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> pd.DataFrame:
        """Nettoyage automatique complet piloté par LLM"""
        df_cleaned = df.copy()
        
        logger.info("Début du nettoyage intelligent 100% LLM")
        
        for col_name, profile in profiles.items():
            df_cleaned = self._clean_column_with_llm(df_cleaned, col_name, profile)
        
        self._log_cleaning_summary(df, df_cleaned)
        return df_cleaned
    
    def _clean_column_with_llm(self, df: pd.DataFrame, column: str, profile: ColumnProfile) -> pd.DataFrame:
        """Nettoyage entièrement décidé par le LLM"""
        original_count = len(df)
        
        # Échantillon de données pour le LLM
        sample_data = df[column].dropna().head(20).tolist()
        
        # Le LLM décide TOUT
        cleaning_strategy = self.llama.get_cleaning_strategy(profile, sample_data)
        
        print(f"LLM Strategy pour {column}: {cleaning_strategy.get('data_category')} -> {cleaning_strategy.get('business_logic')}")
        
        # Application pure des décisions LLM
        df = self._apply_llm_strategy(df, column, cleaning_strategy, profile)
        
        # Log
        rows_removed = original_count - len(df)
        self.cleaning_log.append({
            'column': column,
            'llm_category': cleaning_strategy.get('data_category'),
            'rows_removed': rows_removed,
            'llm_logic': cleaning_strategy.get('business_logic'),
            'actions_taken': cleaning_strategy.get('cleaning_actions', [])
        })
        
        return df
    
    def _apply_llm_strategy(self, df: pd.DataFrame, column: str, strategy: Dict, profile: ColumnProfile) -> pd.DataFrame:
        """Application directe de la stratégie LLM sans interprétation"""
        
        data_category = strategy.get('data_category', 'unknown')
        removal_rules = strategy.get('removal_rules', [])
        cleaning_actions = strategy.get('cleaning_actions', [])
        
        print(f"  Catégorie LLM: {data_category}")
        print(f"  Actions: {cleaning_actions}")
        print(f"  Règles suppression: {removal_rules}")
        
        # Application des règles de suppression basées sur les décisions LLM
        for rule in removal_rules:
            df = self._apply_removal_rule(df, column, rule, data_category)
        
        # Application des actions de nettoyage
        for action in cleaning_actions:
            df = self._apply_cleaning_action(df, column, action, data_category)
        
        return df
    
    def _apply_removal_rule(self, df: pd.DataFrame, column: str, rule: str, category: str) -> pd.DataFrame:
        """Application d'une règle de suppression spécifique"""
        initial_count = len(df)
        
        rule_lower = rule.lower()
        
        # Règles génériques basées sur la logique métier
        if 'null' in rule_lower or 'manquant' in rule_lower:
            df = df.dropna(subset=[column])
        
        elif 'négatif' in rule_lower or 'negative' in rule_lower:
            if df[column].dtype in ['int64', 'float64']:
                df = df[df[column] >= 0]
        
        elif 'zéro' in rule_lower or 'zero' in rule_lower:
            if df[column].dtype in ['int64', 'float64']:
                df = df[df[column] != 0]
        
        elif 'annulation' in rule_lower or 'cancel' in rule_lower:
            if df[column].dtype == 'object':
                # Détection flexible des codes d'annulation
                mask = df[column].astype(str).str.upper().str.startswith('C')
                df = df[~mask]
        
        elif 'vide' in rule_lower or 'empty' in rule_lower:
            if df[column].dtype == 'object':
                mask = (df[column].astype(str).str.strip() == '') | (df[column].astype(str) == 'nan')
                df = df[~mask]
        
        elif 'futur' in rule_lower or 'future' in rule_lower:
            if 'datetime' in str(df[column].dtype):
                future_mask = df[column] > pd.Timestamp.now() + pd.DateOffset(years=1)
                df = df[~future_mask]
        
        removed = initial_count - len(df)
        if removed > 0:
            print(f"    Règle '{rule}': supprimé {removed} lignes")
        
        return df
    
    def _apply_cleaning_action(self, df: pd.DataFrame, column: str, action: str, category: str) -> pd.DataFrame:
        """Application d'une action de nettoyage spécifique"""
        
        action_lower = action.lower()
        
        # Actions de transformation (sans suppression)
        if 'normaliser' in action_lower or 'normalize' in action_lower:
            if df[column].dtype == 'object':
                df[column] = df[column].astype(str).str.strip().str.title()
        
        elif 'standardiser' in action_lower or 'standardize' in action_lower:
            if 'datetime' in str(df[column].dtype):
                df[column] = pd.to_datetime(df[column], errors='coerce')
        
        elif 'convertir' in action_lower or 'convert' in action_lower:
            if 'date' in action_lower:
                df[column] = pd.to_datetime(df[column], errors='coerce')
        
        # Pour les identifiants: préservation maximale (pas d'actions statistiques)
        if category == 'identifier':
            print(f"    Identifiant détecté: préservation des données")
        
        return df
    
    def _log_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        """Log du résumé de nettoyage"""
        logger.info("=== RÉSUMÉ DU NETTOYAGE ===")
        logger.info(f"Lignes originales: {len(original_df)}")
        logger.info(f"Lignes après nettoyage: {len(cleaned_df)}")
        logger.info(f"Colonnes: {len(cleaned_df.columns)}")
        
        for log_entry in self.cleaning_log:
            logger.info(f"Colonne {log_entry['column']}: {log_entry['rows_removed']} lignes supprimées")
            logger.info(f"  Catégorie LLM: {log_entry['llm_category']}")
            logger.info(f"  Logique: {log_entry['llm_logic']}")
            logger.info(f"  Actions: {log_entry['actions_taken']}")

class ValidationEngine:
    """Moteur de validation et scoring de la qualité"""
    
    def validate_cleaning(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, float]:
        """Validation et scoring de la qualité après nettoyage"""
        
        metrics = {}
        
        # Taux de complétude
        original_completeness = (1 - original_df.isnull().sum().sum() / (len(original_df) * len(original_df.columns)))
        cleaned_completeness = (1 - cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns)))
        
        metrics['completeness_improvement'] = cleaned_completeness - original_completeness
        metrics['final_completeness'] = cleaned_completeness
        
        # Taux de conservation des données
        metrics['data_retention'] = len(cleaned_df) / len(original_df)
        
        # Score de qualité global (0-100)
        quality_score = (
            cleaned_completeness * 40 +  # 40% pour la complétude
            metrics['data_retention'] * 30 +  # 30% pour la conservation
            0.3 * 30  # 30% pour la cohérence (simplifié)
        )
        
        metrics['overall_quality_score'] = min(quality_score * 100, 100)
        
        return metrics

class DataCleaningAgent:
    """Agent principal de nettoyage automatique des données"""
    
    def __init__(self, azure_api_key: str):
        self.llama = LlamaIntegration(azure_api_key)
        self.profiler = DataProfiler(self.llama)
        self.cleaner = SmartCleaner(self.llama)
        self.validator = ValidationEngine()
    
    def process_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Traitement complet d'un fichier"""
        
        # Chargement du fichier
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non supporté")
        
        logger.info(f"Fichier chargé: {file_path}")
        logger.info(f"Dimensions: {df.shape}")
        
        # Profiling
        profiles = self.profiler.profile_dataset(df)
        
        # Nettoyage
        df_cleaned = self.cleaner.clean_dataset(df, profiles)
        
        # Validation
        validation_metrics = self.validator.validate_cleaning(df, df_cleaned)
        
        # Rapport complet
        report = {
            'original_shape': df.shape,
            'cleaned_shape': df_cleaned.shape,
            'column_profiles': profiles,
            'cleaning_log': self.cleaner.cleaning_log,
            'validation_metrics': validation_metrics,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return df_cleaned, report
    
    def generate_report(self, report: Dict[str, Any]) -> str:
        """Génère un rapport textuel détaillé"""
        
        report_text = f"""
=== RAPPORT DE NETTOYAGE AUTOMATIQUE ===
Date: {report['processing_timestamp']}

RÉSUMÉ:
- Dimensions originales: {report['original_shape']}
- Dimensions après nettoyage: {report['cleaned_shape']}
- Score de qualité final: {report['validation_metrics']['overall_quality_score']:.1f}/100

ANALYSE PAR COLONNE:
"""
        
        for col_name, profile in report['column_profiles'].items():
            report_text += f"""
{col_name}:
  - Type: {profile.data_type.value}
  - Contexte métier: {profile.business_context} (confiance: {profile.confidence_score:.2f})
  - Valeurs manquantes: {profile.missing_percentage:.1f}%
  - Problèmes détectés: {[issue.value for issue in profile.quality_issues]}
  - Actions recommandées: {profile.recommended_actions}
"""
        
        report_text += f"""
MÉTRIQUES DE VALIDATION:
- Amélioration de complétude: {report['validation_metrics']['completeness_improvement']:.3f}
- Taux de conservation des données: {report['validation_metrics']['data_retention']:.3f}
- Complétude finale: {report['validation_metrics']['final_completeness']:.3f}
"""
        
        return report_text

# Configuration et chargement des secrets
def load_config():
    """Charge la configuration depuis le fichier secrets"""
    try:
        import secrets_config
        return secrets_config.AZURE_API_KEY
    except ImportError:
        print("Fichier 'secrets_config.py' non trouvé!")
        print("Créez un fichier 'secrets_config.py' avec:")
        print("AZURE_API_KEY = 'votre_cle_api_ici'")
        return None
    except AttributeError:
        print("Variable 'AZURE_API_KEY' non trouvée dans secrets_config.py")
        return None

# Script principal pour traitement du fichier
def main():
    """Traite le fichier de données avec l'agent de nettoyage"""
    
    # Vérification des dépendances Azure
    try:
        from azure.ai.inference import ChatCompletionsClient
    except ImportError:
        print("Azure AI Inference non installé!")
        print("Installez avec: pip install azure-ai-inference")
        return
    
    # Configuration
    DATA_PATH = r"C:\Users\Selim\OneDrive\Bureau\data science agent\data\data.xlsx"
    
    # Chargement de la clé API depuis le fichier secrets
    API_KEY = load_config()
    if not API_KEY:
        return
    
    try:
        # Initialisation de l'agent
        print("Initialisation de l'Agent Data Scientist...")
        agent = DataCleaningAgent(API_KEY)
        
        # Vérification du fichier
        if not os.path.exists(DATA_PATH):
            print(f"Fichier non trouvé: {DATA_PATH}")
            return
        
        # Traitement du fichier
        print(f"Traitement du fichier: {DATA_PATH}")
        df_cleaned, report = agent.process_file(DATA_PATH)
        
        # Affichage des résultats
        print("\n" + "="*60)
        print("RÉSULTATS DU NETTOYAGE INTELLIGENT")
        print("="*60)
        print(agent.generate_report(report))
        
        # Sauvegarde du fichier nettoyé
        output_path = DATA_PATH.replace('.xlsx', '_cleaned.xlsx')
        df_cleaned.to_excel(output_path, index=False)
        print(f"Fichier nettoyé sauvegardé: {output_path}")
        
        # Sauvegarde du rapport
        report_path = DATA_PATH.replace('.xlsx', '_cleaning_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(agent.generate_report(report))
        print(f"Rapport sauvegardé: {report_path}")
        
        print(f"\nTraitement terminé avec succès!")
        print(f"Score de qualité final: {report['validation_metrics']['overall_quality_score']:.1f}/100")
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        logger.error(f"Erreur dans main(): {e}", exc_info=True)

if __name__ == "__main__":
    main()