"""
Test de la configuration Power BI
"""
import pandas as pd
from config.powerbi_config import powerbi_config
from config.powerbi_manager import powerbi_manager

def test_powerbi_configuration():
    """
    POURQUOI ce test ?
    
    Power BI est plus complexe que la DB car il nécessite :
    1. Une App Azure AD enregistrée
    2. Des permissions spécifiques  
    3. Des credentials corrects
    
    Ce test vérifie chaque étape pour diagnostiquer les problèmes.
    """
    print("🔌 Test de la configuration Power BI...\n")
    
    # 1. Vérifier les variables d'environnement
    print("1️⃣ Validation de la configuration...")
    if not powerbi_config.validate_configuration():
        print("❌ Configuration incomplète. Vérifiez votre fichier .env")
        return False
    
    # 2. Test d'authentification 
    print("\n2️⃣ Test d'authentification...")
    token = powerbi_config.get_access_token()
    if not token:
        print("❌ Impossible de s'authentifier à Power BI")
        print("💡 Vérifiez vos credentials Azure AD")
        return False
    
    # 3. Test de connexion API
    print("\n3️⃣ Test de connexion API...")
    if not powerbi_config.test_connection():
        print("❌ Impossible de se connecter à l'API Power BI")
        return False
    
    # 4. Test de workspace
    print("\n4️⃣ Test de workspace...")
    workspace_id = powerbi_manager.setup_workspace("Test Autonomous Agent")
    if not workspace_id:
        print("❌ Impossible de créer/accéder au workspace")
        return False
    
    print(f"✅ Workspace configuré: {workspace_id}")
    
    print("\n🎉 Configuration Power BI complètement fonctionnelle !")
    return True

def test_powerbi_with_sample_data():
    """
    Test complet avec des données d'exemple
    
    POURQUOI ce test ?
    Simuler le workflow complet de notre agent :
    1. Créer des données d'exemple
    2. Faire une "fausse" analyse  
    3. Envoyer vers Power BI
    4. Créer un dataset
    """
    print("\n📊 Test avec données d'exemple...")
    
    # Créer un DataFrame d'exemple
    sample_data = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'sales': [1000, 1200, 1100, 1300, 1250],
        'profit': [100, 150, 120, 180, 160],
        'region': ['North', 'South', 'North', 'West', 'East']
    })
    
    # Simuler des résultats d'analyse
    analysis_results = {
        'quality_score': 8.5,
        'missing_percentage': 0.0,
        'total_rows': len(sample_data),
        'total_columns': len(sample_data.columns),
        'analysis_type': 'test'
    }
    
    print(f"📋 Données d'exemple créées: {sample_data.shape}")
    print("Colonnes:", list(sample_data.columns))
    
    # Tenter de créer un dataset Power BI
    success = powerbi_manager.create_dataset_from_analysis(
        dataset_name="Test_Dataset_Agent",
        analysis_results=analysis_results,
        dataframe=sample_data
    )
    
    if success:
        print("✅ Dataset Power BI créé avec succès !")
    else:
        print("❌ Erreur lors de la création du dataset")
        return False
    
    return True

def run_powerbi_tests():
    """Exécuter tous les tests Power BI"""
    print("🚀 Tests Power BI - Agent Data Scientist\n")
    
    # Test de base
    if not test_powerbi_configuration():
        print("\n❌ Tests de base échoués")
        print("\n💡 AIDE POUR CONFIGURER POWER BI:")
        print("1. Aller sur https://portal.azure.com")
        print("2. Azure Active Directory > App registrations")
        print("3. Créer une nouvelle app")
        print("4. Copier Client ID, Tenant ID")
        print("5. Créer un Client Secret")
        print("6. Ajouter les permissions Power BI")
        print("7. Mettre à jour le fichier .env")
        return
    
    # Test avancé avec données
    test_powerbi_with_sample_data()

if __name__ == "__main__":
    run_powerbi_tests()