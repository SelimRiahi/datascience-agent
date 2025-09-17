"""
Test de la configuration de base de données
"""
from config.database import db_config
from database.models import Dataset, AnalysisResult, MLModel

def test_database_setup():
    """Tester la configuration complète de la base de données"""
    print("🗄️ Test de la configuration de base de données...\n")
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion...")
    if not db_config.test_connection():
        print("❌ Impossible de se connecter à la base de données")
        return False
    
    # 2. Création des tables
    print("\n2️⃣ Création des tables...")
    if not db_config.create_tables():
        print("❌ Impossible de créer les tables")
        return False
    
    # 3. Test d'insertion de données
    print("\n3️⃣ Test d'insertion de données...")
    try:
        session = db_config.get_session()
        
        # Créer un dataset de test
        test_dataset = Dataset(
            name="Dataset de Test",
            filename="test.csv",
            file_path="/storage/test.csv",
            rows_count=1000,
            columns_count=5,
            file_size_mb=0.5
        )
        
        session.add(test_dataset)
        session.commit()
        
        print("✅ Dataset de test inséré avec succès")
        print(f"   ID: {test_dataset.id}")
        print(f"   Nom: {test_dataset.name}")
        print(f"   Créé le: {test_dataset.created_at}")
        
        # Test de lecture
        datasets = session.query(Dataset).all()
        print(f"\n📊 Nombre total de datasets: {len(datasets)}")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Erreur lors du test d'insertion: {e}")
        return False
    
    print("\n🎉 Configuration de base de données complètement fonctionnelle !")
    return True

if __name__ == "__main__":
    test_database_setup()