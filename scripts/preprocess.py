import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # Drop rows with missing Grade (if any)
    df = df.dropna(subset=['Grade'])
    
    # Convert Age_at_diagnosis to numerical (extract years)
    df['Age'] = df['Age_at_diagnosis'].str.extract(r'(\d+)').astype(float)
    
    # Encode binary mutation status (1=MUTATED, 0=NOT_MUTATED)
    mutation_cols = df.columns[7:]  # All columns after 'PDGFRA'
    for col in mutation_cols:
        df[col] = df[col].map({'MUTATED': 1, 'NOT_MUTATED': 0})
    
    # Encode categorical variables (Gender, Race)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, columns=['Race'], drop_first=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(
        input_path=r"C:\Users\Tejas\OneDrive\Desktop\glioma_grading\data\glicomaxl\TCGA_GBM_LGG_Mutations_all.csv",
        output_path=r"C:\Users\Tejas\OneDrive\Desktop\glioma_grading\data\glicomaxl\cleaned_data.csv"
    )