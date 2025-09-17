import pandas as pd
import numpy as np

def validate_cleaning_results(original_file_path, cleaned_file_path):
    """Validate manually what the LLM agent actually removed"""
    
    print("🔍 VALIDATION MANUELLE DES RÉSULTATS DE NETTOYAGE")
    print("=" * 60)
    
    # Load both files
    df_original = pd.read_excel(original_file_path)
    df_cleaned = pd.read_excel(cleaned_file_path)
    
    print(f"📊 Données originales: {len(df_original)} lignes")
    print(f"📊 Données nettoyées: {len(df_cleaned)} lignes")
    print(f"📊 Différence: {len(df_original) - len(df_cleaned)} lignes supprimées")
    print()
    
    # Check Description column specifically
    print("🔍 ANALYSE DE LA COLONNE DESCRIPTION:")
    print("-" * 40)
    
    # Count different types of missing/empty values in original
    desc_col = df_original['Description']
    
    # Various ways to check for empty/missing
    null_count = desc_col.isnull().sum()
    empty_string_count = (desc_col == '').sum()
    whitespace_only_count = desc_col.astype(str).str.strip().eq('').sum() - empty_string_count
    nan_string_count = desc_col.astype(str).str.lower().eq('nan').sum()
    
    print(f"Valeurs NULL (pd.isnull()): {null_count}")
    print(f"Chaînes vides (''): {empty_string_count}")
    print(f"Espaces uniquement: {whitespace_only_count}")
    print(f"Chaînes 'nan': {nan_string_count}")
    
    # Total problematic descriptions
    problematic_mask = (
        desc_col.isnull() | 
        (desc_col == '') | 
        (desc_col.astype(str).str.strip() == '') |
        (desc_col.astype(str).str.lower() == 'nan')
    )
    
    total_problematic = problematic_mask.sum()
    print(f"📍 TOTAL problématique: {total_problematic}")
    print()
    
    # Check what was actually removed
    print("🔍 VÉRIFICATION DE CE QUI A ÉTÉ SUPPRIMÉ:")
    print("-" * 45)
    
    if total_problematic == (len(df_original) - len(df_cleaned)):
        print("✅ CORRESPONDANCE PARFAITE!")
        print(f"   LLM a supprimé exactement les {total_problematic} descriptions vides/manquantes")
    else:
        print("❌ DIFFÉRENCE DÉTECTÉE!")
        print(f"   Descriptions problématiques: {total_problematic}")
        print(f"   Lignes supprimées par LLM: {len(df_original) - len(df_cleaned)}")
        print("   Investigation supplémentaire nécessaire...")
    
    print()
    
    # Show sample of problematic descriptions
    print("📋 ÉCHANTILLON DES DESCRIPTIONS PROBLÉMATIQUES:")
    print("-" * 50)
    
    problematic_rows = df_original[problematic_mask]
    if len(problematic_rows) > 0:
        sample_size = min(10, len(problematic_rows))
        print(f"Affichage des {sample_size} premières sur {len(problematic_rows)}:")
        
        for i, (idx, row) in enumerate(problematic_rows.head(sample_size).iterrows()):
            desc_value = row['Description']
            desc_repr = repr(desc_value)  # Shows actual representation
            print(f"  {i+1:2d}. Index {idx}: {desc_repr}")
    
    print()
    
    # Check other columns for completeness
    print("🔍 VÉRIFICATION AUTRES COLONNES:")
    print("-" * 35)
    
    for col in df_original.columns:
        if col != 'Description':
            original_nulls = df_original[col].isnull().sum()
            cleaned_nulls = df_cleaned[col].isnull().sum() if col in df_cleaned.columns else 0
            
            if original_nulls > 0:
                print(f"{col}: {original_nulls} nulls originaux → {cleaned_nulls} nulls nettoyés")
    
    print()
    
    # Final verification
    print("🎯 CONCLUSION:")
    print("-" * 12)
    
    if total_problematic == 1454:
        print("✅ LLM était CORRECT: exactement 1454 descriptions vides à supprimer")
    else:
        print(f"⚠️  Différence: LLM dit 1454, mais nous trouvons {total_problematic}")
    
    return {
        'original_rows': len(df_original),
        'cleaned_rows': len(df_cleaned),
        'rows_removed': len(df_original) - len(df_cleaned),
        'problematic_descriptions': total_problematic,
        'llm_accurate': total_problematic == 1454
    }

# Script de validation
if __name__ == "__main__":
    # Paths to your files
    original_path = r"C:\Users\Selim\OneDrive\Bureau\data science agent\data\data.xlsx"
    cleaned_path = r"C:\Users\Selim\OneDrive\Bureau\data science agent\data\data_cleaned.xlsx"
    
    try:
        results = validate_cleaning_results(original_path, cleaned_path)
        
        print("\n" + "=" * 60)
        print("📈 RÉSUMÉ DE VALIDATION")
        print("=" * 60)
        print(f"Exactitude du LLM: {'OUI' if results['llm_accurate'] else 'NON'}")
        print(f"Lignes supprimées: {results['rows_removed']}")
        print(f"Descriptions problématiques détectées: {results['problematic_descriptions']}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")