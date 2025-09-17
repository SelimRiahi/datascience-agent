"""
Configuration Power BI API pour l'Agent Data Scientist
"""
import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()

class PowerBIConfig:
    """Configuration et gestion de l'API Power BI"""
    
    def __init__(self):
        # Paramètres d'authentification Azure AD
        self.client_id = os.getenv('POWERBI_CLIENT_ID')
        self.client_secret = os.getenv('POWERBI_CLIENT_SECRET') 
        self.tenant_id = os.getenv('POWERBI_TENANT_ID')
        self.username = os.getenv('POWERBI_USERNAME')
        self.password = os.getenv('POWERBI_PASSWORD')
        
        # URLs de base Power BI
        self.authority_url = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.powerbi_api_url = "https://api.powerbi.com/v1.0/myorg"
        self.scope = ["https://analysis.windows.net/powerbi/api/.default"]
        
        # Token d'accès (sera généré)
        self.access_token = None
    
    def get_access_token(self) -> Optional[str]:
        """
        POURQUOI cette fonction ?
        
        Power BI utilise OAuth2 pour la sécurité. Chaque appel API nécessite
        un token d'accès valide. Cette fonction :
        1. S'authentifie avec nos credentials
        2. Récupère un token temporaire (1h de validité)  
        3. Permet d'appeler l'API Power BI
        """
        try:
            # URL pour obtenir le token
            token_url = f"{self.authority_url}/oauth2/v2.0/token"
            
            # Données pour l'authentification
            data = {
                'grant_type': 'password',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'username': self.username,
                'password': self.password,
                'scope': ' '.join(self.scope)
            }
            
            # Appel API pour récupérer le token
            response = requests.post(token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                print("✅ Token Power BI obtenu avec succès")
                return self.access_token
            else:
                print(f"❌ Erreur d'authentification Power BI: {response.status_code}")
                print(f"Détail: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lors de l'obtention du token: {e}")
            return None
    
    def get_headers(self) -> Dict[str, str]:
        """
        POURQUOI cette fonction ?
        
        Chaque appel à l'API Power BI doit inclure :
        - Le token d'authentification dans le header
        - Le type de contenu JSON
        Cette fonction standardise ces headers pour tous nos appels API.
        """
        if not self.access_token:
            self.get_access_token()
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        POURQUOI ce test ?
        
        Avant d'utiliser Power BI, on doit vérifier :
        1. Nos credentials sont corrects
        2. On peut s'authentifier  
        3. L'API répond correctement
        
        Ce test appelle l'endpoint le plus simple : récupérer la liste des workspaces
        """
        try:
            headers = self.get_headers()
            if not headers.get('Authorization'):
                return False
            
            # Test simple : récupérer les workspaces
            url = f"{self.powerbi_api_url}/groups"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                workspaces = response.json().get('value', [])
                print(f"✅ Connexion Power BI réussie!")
                print(f"📊 Workspaces disponibles: {len(workspaces)}")
                
                # Afficher les workspaces
                for ws in workspaces[:3]:  # Afficher seulement les 3 premiers
                    print(f"   - {ws.get('name', 'Sans nom')}")
                
                return True
            else:
                print(f"❌ Erreur API Power BI: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de connexion Power BI: {e}")
            return False
    
    def get_workspaces(self) -> list:
        """Récupérer tous les workspaces Power BI"""
        headers = self.get_headers()
        url = f"{self.powerbi_api_url}/groups"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json().get('value', [])
            return []
        except Exception as e:
            print(f"Erreur lors de la récupération des workspaces: {e}")
            return []
    
    def validate_configuration(self) -> bool:
        """
        POURQUOI cette validation ?
        
        Avant de commencer à utiliser Power BI, on doit s'assurer que
        toutes les variables d'environnement nécessaires sont définies.
        Cela évite les erreurs surprises plus tard.
        """
        required_vars = [
            ('POWERBI_CLIENT_ID', self.client_id),
            ('POWERBI_CLIENT_SECRET', self.client_secret), 
            ('POWERBI_TENANT_ID', self.tenant_id),
            ('POWERBI_USERNAME', self.username),
            ('POWERBI_PASSWORD', self.password)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print("❌ Variables d'environnement manquantes pour Power BI:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\n💡 Ajoute ces variables dans ton fichier .env")
            return False
        
        print("✅ Configuration Power BI complète")
        return True

# Instance globale
powerbi_config = PowerBIConfig()