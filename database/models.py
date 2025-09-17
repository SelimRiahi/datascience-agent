"""
Modèles de base de données pour l'Agent Data Scientist
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.sql import func
from config.database import db_config

Base = db_config.Base

class Dataset(Base):
    """Modèle pour stocker les informations des datasets"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Métadonnées du dataset
    rows_count = Column(Integer)
    columns_count = Column(Integer)
    file_size_mb = Column(Float)
    
    # Informations temporelles
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Statut du traitement
    is_processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', rows={self.rows_count})>"

class AnalysisResult(Base):
    """Modèle pour stocker les résultats d'analyses"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False)  # Foreign key vers Dataset
    analysis_type = Column(String(50), nullable=False)  # 'cleaning', 'ml', 'visualization'
    
    # Résultats en JSON
    results_json = Column(Text)  # Stockage JSON des résultats
    metrics_json = Column(Text)  # Métriques de performance
    
    # Métadonnées
    execution_time_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AnalysisResult(dataset_id={self.dataset_id}, type='{self.analysis_type}')>"

class MLModel(Base):
    """Modèle pour stocker les informations des modèles ML"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'classification', 'regression'
    
    # Métriques de performance
    accuracy_score = Column(Float)
    training_time = Column(Float)
    
    # Chemin vers le modèle sauvegardé
    model_path = Column(String(500))
    
    # Paramètres du modèle (JSON)
    parameters_json = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<MLModel(name='{self.model_name}', accuracy={self.accuracy_score})>"