"""
Gestionnaire Power BI pour automatiser la création de dashboards
"""
import pandas as pd
import json
import requests
from typing import Dict, Any, Optional, List
from config.powerbi_config import powerbi_config

class PowerBIManager:
    """
    POURQUOI cette classe ?
    
    Notre Agent Data Scientist va générer des analyses, mais les utilisateurs
    business ont besoin de dashboards interactifs. Cette classe :
    
    1. Prend les résultats de nos analyses Python
    2. Les formate pour Power BI  
    3. Crée automatiquement des dashboards
    4. Les publie sur Power BI Service
    
    = AUTOMATION COMPLETE de Python vers Power BI !
    """
    
    def __init__(self):
        self.config = powerbi_config
        self.default_workspace_id = None
    
    def setup_workspace(self, workspace_name: str = "Autonomous Data Scientist") -> Optional[str]:
        """
        POURQUOI un workspace dédié ?
        
        Power BI organise tout en "workspaces" (espaces de travail).
        Notre agent va créer plein de dashboards, donc on veut :
        1. Un workspace dédié pour s'organiser
        2. Éviter de polluer les autres workspaces
        3. Permissions centralisées
        """
        try:
            headers = self.config.get_headers()
            
            # Vérifier si le workspace existe déjà
            workspaces = self.config.get_workspaces()
            for ws in workspaces:
                if ws['name'] == workspace_name:
                    self.default_workspace_id = ws['id']
                    print(f"✅ Workspace '{workspace_name}' trouvé: {ws['id']}")
                    return ws['id']
            
            # Créer le workspace s'il n'existe pas
            url = f"{self.config.powerbi_api_url}/groups"
            data = {"name": workspace_name}
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                workspace_id = response.json()['id']
                self.default_workspace_id = workspace_id
                print(f"✅ Workspace '{workspace_name}' créé: {workspace_id}")
                return workspace_id
            else:
                print(f"❌ Erreur création workspace: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur setup workspace: {e}")
            return None
    
    def create_dataset_from_analysis(self, 
                                   dataset_name: str,
                                   analysis_results: Dict[str, Any],
                                   dataframe: pd.DataFrame) -> bool:
        """
        POURQUOI cette fonction ?
        
        Quand notre agent analyse un CSV, il génère :
        - Des statistiques (moyennes, médianes, etc.)
        - Des résultats de nettoyage
        - Des métriques de qualité
        
        Cette fonction transforme tout ça en "dataset Power BI" 
        que les dashboards pourront utiliser.
        """
        try:
            if not self.default_workspace_id:
                print("❌ Aucun workspace configuré")
                return False
            
            headers = self.config.get_headers()
            
            # Créer la structure du dataset pour Power BI
            dataset_schema = self._create_dataset_schema(dataset_name, analysis_results, dataframe)
            
            # API call pour créer le dataset
            url = f"{self.config.powerbi_api_url}/groups/{self.default_workspace_id}/datasets"
            
            response = requests.post(url, headers=headers, json=dataset_schema)
            
            if response.status_code == 201:
                dataset_id = response.json()['id']
                print(f"✅ Dataset Power BI créé: {dataset_id}")
                
                # Ajouter les données au dataset
                return self._upload_data_to_dataset(dataset_id, analysis_results, dataframe)
            else:
                print(f"❌ Erreur création dataset: {response.status_code}")
                print(f"Détail: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur création dataset: {e}")
            return False
    
    def _create_dataset_schema(self, name: str, analysis: Dict, df: pd.DataFrame) -> Dict:
        """
        POURQUOI définir un schéma ?
        
        Power BI a besoin de connaître la structure des données :
        - Quelles colonnes ?
        - Quels types de données ?
        - Quelles relations ?
        
        Cette fonction traduit notre DataFrame pandas en schéma Power BI.
        """
        
        # Analyser les types de colonnes du DataFrame
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Mapper les types pandas vers Power BI
            if 'int' in dtype:
                pbi_type = 'Int64'
            elif 'float' in dtype:
                pbi_type = 'Double'
            elif 'datetime' in dtype:
                pbi_type = 'DateTime'
            elif 'bool' in dtype:
                pbi_type = 'Boolean'
            else:
                pbi_type = 'String'
            
            columns.append({
                "name": col,
                "dataType": pbi_type
            })
        
        # Ajouter des colonnes pour les métrics d'analyse
        analysis_columns = [
            {"name": "quality_score", "dataType": "Double"},
            {"name": "missing_percentage", "dataType": "Double"},
            {"name": "analysis_date", "dataType": "DateTime"}
        ]
        
        return {
            "name": name,
            "tables": [
                {
                    "name": "MainData",
                    "columns": columns
                },
                {
                    "name": "AnalysisMetrics", 
                    "columns": analysis_columns
                }
            ]
        }
    
    def _upload_data_to_dataset(self, dataset_id: str, analysis: Dict, df: pd.DataFrame) -> bool:
        """
        POURQUOI uploader les données ?
        
        Créer un dataset vide ne sert à rien ! Il faut :
        1. Envoyer nos données pandas vers Power BI
        2. Formater correctement pour l'API
        3. Gérer les gros volumes (chunking si nécessaire)
        """
        try:
            headers = self.config.get_headers()
            
            # Limiter à 1000 lignes pour le test (API limit)
            df_sample = df.head(1000) if len(df) > 1000 else df
            
            # Convertir en format JSON pour Power BI
            data_rows = df_sample.to_dict('records')
            
            # URL pour ajouter des données
            url = f"{self.config.powerbi_api_url}/datasets/{dataset_id}/tables/MainData/rows"
            
            # Chunking pour les gros datasets
            chunk_size = 100  # Power BI limite à ~200 lignes par appel
            
            for i in range(0, len(data_rows), chunk_size):
                chunk = data_rows[i:i + chunk_size]
                data = {"rows": chunk}
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code != 200:
                    print(f"❌ Erreur upload chunk {i}: {response.status_code}")
                    return False
                
                print(f"✅ Chunk {i//chunk_size + 1} uploadé")
            
            print(f"✅ {len(data_rows)} lignes uploadées vers Power BI")
            return True
            
        except Exception as e:
            print(f"❌ Erreur upload données: {e}")
            return False
    
    def create_auto_dashboard(self, dataset_id: str, analysis_results: Dict) -> Optional[str]:
        """
        POURQUOI un dashboard automatique ?
        
        Le but final : l'utilisateur upload un CSV, notre agent :
        1. Analyse les données
        2. Crée automatiquement un dashboard Power BI
        3. L'utilisateur a immédiatement des visualisations pro !
        
        Cette fonction génère un dashboard basé sur l'analyse.
        """
        # Note: L'auto-création de dashboards Power BI via API est limitée
        # On peut créer le dataset, mais les visuels nécessitent Power BI Desktop
        
        print("📊 Dataset créé avec succès dans Power BI!")
        print("💡 Prochaines étapes:")
        print("   1. Aller sur app.powerbi.com")  
        print(f"   2. Trouver le dataset: {dataset_id}")
        print("   3. Créer un rapport avec les visuels")
        print("   4. Publier le dashboard")
        
        return dataset_id

# Instance globale
powerbi_manager = PowerBIManager()