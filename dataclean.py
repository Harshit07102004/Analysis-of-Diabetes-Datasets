import pandas as pd
import numpy as np
import re
from collections import Counter

def clean_diabetic_data(file_path):
    """
    Comprehensive cleaning of diabetic_data.csv for machine learning preparation
    """
    
    print("Loading the dataset...")
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # 1. Handle Missing Values - Replace '?' with NaN
    print("\n1. Handling missing values...")
    df = df.replace('?', np.nan)
    print("Replaced all '?' occurrences with NaN")
    
    # Display missing value counts before cleaning
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    print("\nMissing value percentages before cleaning:")
    for col in missing_counts[missing_counts > 0].index:
        print(f"{col}: {missing_percentages[col]:.2f}%")
    
    # 2. Drop High-Missing-Value Columns
    print("\n2. Dropping high-missing-value columns...")
    columns_to_drop_missing = ['weight', 'medical_specialty', 'payer_code']
    df = df.drop(columns=columns_to_drop_missing)
    print(f"Dropped columns: {columns_to_drop_missing}")
    
    # 3. Address Multiple Encounters - Keep only first encounter per patient
    print("\n3. Addressing multiple encounters...")
    print(f"Shape before removing duplicate patients: {df.shape}")
    
    # Keep only the first encounter for each unique patient
    df_deduplicated = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    removed_encounters = len(df) - len(df_deduplicated)
    df = df_deduplicated
    print(f"Removed {removed_encounters} duplicate patient encounters")
    print(f"Shape after deduplication: {df.shape}")
    
    # 4. Drop Identifier Columns
    print("\n4. Dropping identifier columns...")
    identifier_columns = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=identifier_columns)
    print(f"Dropped identifier columns: {identifier_columns}")
    print(f"Dataset shape after dropping columns: {df.shape}")
    
    # 5. Clean Categorical Features
    print("\n5. Cleaning categorical features...")
    
    # Clean gender column
    print(f"Gender value counts before cleaning:")
    print(df['gender'].value_counts(dropna=False))
    
    # Remove rows with 'Unknown/Invalid' gender
    df = df[df['gender'] != 'Unknown/Invalid']
    print(f"Removed rows with 'Unknown/Invalid' gender. New shape: {df.shape}")
    
    # Clean race column - impute missing values with mode
    race_mode = df['race'].mode()[0]
    df['race'] = df['race'].fillna(race_mode)
    print(f"Imputed missing race values with mode: '{race_mode}'")
    
    # 6. Group Diagnosis Codes
    print("\n6. Grouping diagnosis codes...")
    
    def categorize_diagnosis(diag_code):
        """
        Categorize ICD-9 diagnosis codes into broader categories
        """
        if pd.isna(diag_code) or diag_code == '':
            return 'Unknown'
        
        # Convert to string and extract numeric part
        diag_str = str(diag_code)
        
        # Extract numeric part using regex
        numeric_match = re.search(r'(\d+)', diag_str)
        if not numeric_match:
            return 'Unknown'
        
        try:
            code = int(numeric_match.group(1))
        except:
            return 'Unknown'
        
        # ICD-9 code ranges for different categories
        if 1 <= code <= 139:
            return 'Infectious_Parasitic'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        elif 240 <= code <= 279:
            return 'Endocrine_Metabolic'
        elif 280 <= code <= 289:
            return 'Blood'
        elif 290 <= code <= 319:
            return 'Mental'
        elif 320 <= code <= 389:
            return 'Nervous_System'
        elif 390 <= code <= 459:
            return 'Circulatory'
        elif 460 <= code <= 519:
            return 'Respiratory'
        elif 520 <= code <= 579:
            return 'Digestive'
        elif 580 <= code <= 629:
            return 'Genitourinary'
        elif 630 <= code <= 679:
            return 'Pregnancy'
        elif 680 <= code <= 709:
            return 'Skin'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif 740 <= code <= 759:
            return 'Congenital'
        elif 760 <= code <= 779:
            return 'Perinatal'
        elif 780 <= code <= 799:
            return 'Symptoms'
        elif 800 <= code <= 999:
            return 'Injury_Poisoning'
        elif code == 250 or diag_str.startswith('250'):
            return 'Diabetes'
        else:
            return 'Other'
    
    # Apply categorization to diagnosis columns
    diagnosis_columns = ['diag_1', 'diag_2', 'diag_3']
    for col in diagnosis_columns:
        if col in df.columns:
            print(f"Processing {col}...")
            df[f'{col}_category'] = df[col].apply(categorize_diagnosis)
            # Drop original diagnosis columns
            df = df.drop(columns=[col])
    
    print("Diagnosis codes categorized and original columns dropped")
    
    # 7. Final Checks and Data Type Optimization
    print("\n7. Final checks and optimizations...")
    
    # Check for remaining missing values
    remaining_missing = df.isnull().sum()
    if remaining_missing.sum() > 0:
        print("Remaining missing values:")
        for col in remaining_missing[remaining_missing > 0].index:
            print(f"{col}: {remaining_missing[col]} ({(remaining_missing[col]/len(df)*100):.2f}%)")
    else:
        print("No missing values remaining!")
    
    # Display data types
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Display sample of categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(categorical_columns)}")
    
    for col in categorical_columns:
        unique_values = df[col].nunique()
        print(f"{col}: {unique_values} unique values")
        if unique_values <= 10:
            print(f"  Values: {list(df[col].unique())}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a new CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = r"D:\Project diabeties\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"
    output_file = r"D:\Project diabeties\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data_cleaned.csv"
    
    try:
        # Clean the data
        cleaned_df = clean_diabetic_data(input_file)
        
        # Save the cleaned data
        save_cleaned_data(cleaned_df, output_file)
        
        print("\n" + "="*50)
        print("DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Display final summary
        print(f"\nFinal Summary:")
        print(f"- Final dataset shape: {cleaned_df.shape}")
        print(f"- Total missing values: {cleaned_df.isnull().sum().sum()}")
        print(f"- Categorical columns: {len(cleaned_df.select_dtypes(include=['object']).columns)}")
        print(f"- Numerical columns: {len(cleaned_df.select_dtypes(include=[np.number]).columns)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_file}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data and try again.")