"""
Configuration des bases de données
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class DatabaseConfig:
    """Configuration centralisée pour les bases de données"""
    
    def __init__(self):
        # URL de base de données (par défaut SQLite pour commencer)
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'sqlite:///./autonomous_ds.db'
        )
        
        # Configuration de l'engine SQLAlchemy
        self.engine = create_engine(
            self.database_url,
            echo=True,  # Afficher les requêtes SQL pour debug
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Base pour les modèles
        self.Base = declarative_base()
    
    def get_session(self):
        """Créer une nouvelle session de base de données"""
        return self.SessionLocal()
    
    def test_connection(self):
        """Tester la connexion à la base de données"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                print("✅ Connexion à la base de données réussie!")
                return True
        except Exception as e:
            print(f"❌ Erreur de connexion à la base de données: {e}")
            return False
    
    def create_tables(self):
        """Créer toutes les tables définies dans les modèles"""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            print("✅ Tables créées avec succès!")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la création des tables: {e}")
            return False

# Instance globale
db_config = DatabaseConfig()