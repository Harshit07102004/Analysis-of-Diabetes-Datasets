import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_diabetes_dataset(file_path=r'D:\Project diabeties\PIDD\diabetes.csv'):
    """
    Clean the diabetes dataset by handling missing values and duplicates
    
    Parameters:
    file_path (str): Path to the diabetes.csv file
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    
    # Load the dataset
    print("Loading diabetes dataset...")
    df = pd.read_csv(file_path)
    
    # Display initial dataset info
    print("\n" + "="*50)
    print("INITIAL DATASET SUMMARY")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"Total records: {df.shape[0]}")
    print(f"Total features: {df.shape[1]}")
    
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values (including zeros in specific columns)
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Columns that shouldn't have zero values (biological impossibility)
    zero_check_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("Checking for zero values in biologically impossible columns:")
    zero_counts = {}
    for col in zero_check_columns:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_counts[col] = zero_count
            print(f"{col}: {zero_count} zero values ({zero_count/len(df)*100:.2f}%)")
    
    # Store original data for comparison
    original_df = df.copy()
    
    # Replace zeros with median values
    print("\n" + "="*50)
    print("REPLACING ZEROS WITH MEDIAN VALUES")
    print("="*50)
    
    for col in zero_check_columns:
        if col in df.columns and zero_counts[col] > 0:
            # Calculate median excluding zeros
            median_value = df[df[col] != 0][col].median()
            
            print(f"\n{col}:")
            print(f"  - Zero values found: {zero_counts[col]}")
            print(f"  - Median value (excluding zeros): {median_value:.2f}")
            
            # Replace zeros with median
            df.loc[df[col] == 0, col] = median_value
            print(f"  - Replaced {zero_counts[col]} zero values with {median_value:.2f}")
    
    # Check for duplicate rows
    print("\n" + "="*50)
    print("DUPLICATE ROWS ANALYSIS")
    print("="*50)
    
    duplicates_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    
    if duplicates_count > 0:
        print("Removing duplicate rows...")
        df = df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate rows")
    else:
        print("No duplicate rows found")
    
    # Final dataset summary
    print("\n" + "="*50)
    print("CLEANED DATASET SUMMARY")
    print("="*50)
    
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Rows removed: {original_df.shape[0] - df.shape[0]}")
    
    print("\nFinal statistics:")
    print(df.describe())
    
    # Verify no zeros remain in the specified columns
    print("\nVerification - Zero values after cleaning:")
    for col in zero_check_columns:
        if col in df.columns:
            remaining_zeros = (df[col] == 0).sum()
            print(f"{col}: {remaining_zeros} zero values")
    
    # Check for any remaining missing values
    print("\nMissing values (NaN) check:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")
    
    # Data distribution comparison (before and after)
    print("\n" + "="*50)
    print("DATA DISTRIBUTION COMPARISON")
    print("="*50)
    
    for col in zero_check_columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Original mean: {original_df[col].mean():.2f}")
            print(f"  Cleaned mean: {df[col].mean():.2f}")
            print(f"  Original std: {original_df[col].std():.2f}")
            print(f"  Cleaned std: {df[col].std():.2f}")
    
    return df

def create_visualization(original_df, cleaned_df):
    """
    Create visualizations to show the impact of data cleaning
    """
    zero_check_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    available_cols = [col for col in zero_check_columns if col in original_df.columns]
    
    if len(available_cols) == 0:
        print("No columns available for visualization")
        return
    
    # Create subplots for before/after comparison
    fig, axes = plt.subplots(2, len(available_cols), figsize=(4*len(available_cols), 8))
    if len(available_cols) == 1:
        axes = axes.reshape(2, 1)
    
    for i, col in enumerate(available_cols):
        # Original data
        axes[0, i].hist(original_df[col], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, i].set_title(f'{col} - Original')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        
        # Cleaned data
        axes[1, i].hist(cleaned_df[col], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, i].set_title(f'{col} - Cleaned')
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Alternative function for when you have the DataFrame already loaded
def clean_existing_dataframe(df):
    """
    Clean an already loaded diabetes DataFrame
    
    Parameters:
    df (pd.DataFrame): The diabetes dataset DataFrame
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Columns that shouldn't have zero values
    zero_check_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n" + "="*50)
    print("CLEANING EXISTING DATAFRAME")
    print("="*50)
    
    # Display initial info
    print(f"Original dataset shape: {cleaned_df.shape}")
    
    # Check for zero values
    print("\nZero values found in biological columns:")
    zero_counts = {}
    for col in zero_check_columns:
        if col in cleaned_df.columns:
            zero_count = (cleaned_df[col] == 0).sum()
            zero_counts[col] = zero_count
            print(f"{col}: {zero_count} zero values ({zero_count/len(cleaned_df)*100:.2f}%)")
    
    # Replace zeros with median values
    print("\nReplacing zeros with median values:")
    for col in zero_check_columns:
        if col in cleaned_df.columns and zero_counts[col] > 0:
            # Calculate median excluding zeros
            median_value = cleaned_df[cleaned_df[col] != 0][col].median()
            
            print(f"\n{col}:")
            print(f"  - Zero values found: {zero_counts[col]}")
            print(f"  - Median value (excluding zeros): {median_value:.2f}")
            
            # Replace zeros with median
            cleaned_df.loc[cleaned_df[col] == 0, col] = median_value
            print(f"  - Replaced {zero_counts[col]} zero values with {median_value:.2f}")
    
    # Check for duplicates
    print("\n" + "="*30)
    print("CHECKING FOR DUPLICATES")
    print("="*30)
    
    duplicates_count = cleaned_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    
    if duplicates_count > 0:
        print("Removing duplicate rows...")
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate rows")
    else:
        print("No duplicate rows found")
    
    # Final verification
    print("\n" + "="*30)
    print("CLEANING VERIFICATION")
    print("="*30)
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    # Verify no zeros remain in the specified columns
    print("\nVerification - Zero values after cleaning:")
    for col in zero_check_columns:
        if col in cleaned_df.columns:
            remaining_zeros = (cleaned_df[col] == 0).sum()
            print(f"{col}: {remaining_zeros} zero values")
    
    print("\n✅ Dataset cleaning completed successfully!")
    
    return cleaned_df

# Quick analysis function for specific file path
def analyze_diabetes_data(file_path=r'D:\Project diabeties\PIDD\diabetes.csv'):
    """
    Analyze the diabetes.csv data from specific file path
    """
    try:
        # Read the file from the specified path
        df = pd.read_csv(file_path)
        
        print(f"Successfully loaded diabetes.csv from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        
        # Clean the data
        cleaned_df = clean_existing_dataframe(df)
        
        # Save cleaned data in the same directory
        import os
        output_path = os.path.join(os.path.dirname(file_path), 'diabetes_cleaned.csv')
        cleaned_df.to_csv(output_path, index=False)
        print(f"\n📁 Cleaned dataset saved to: {output_path}")
        
        return df, cleaned_df
        
    except Exception as e:
        print(f"Error reading file from {file_path}: {e}")
        return None, None

# Main execution
if __name__ == "__main__":
    # Set your file path
    diabetes_file_path = r'D:\Project diabeties\PIDD\diabetes.csv'
    
    try:
        print("🔄 Loading diabetes dataset from:", diabetes_file_path)
        
        # Analyze and clean the data
        original_data, cleaned_data = analyze_diabetes_data(diabetes_file_path)
        
        if cleaned_data is not None:
            print("\n" + "="*50)
            print("✅ ANALYSIS COMPLETE - DATA SUCCESSFULLY CLEANED")
            print("="*50)
            print(f"📊 Original dataset: {original_data.shape[0]} rows, {original_data.shape[1]} columns")
            print(f"🧹 Cleaned dataset: {cleaned_data.shape[0]} rows, {cleaned_data.shape[1]} columns")
            print(f"💾 Cleaned file location: D:\\Project diabeties\\PIDD\\diabetes_cleaned.csv")
            
            # Show basic statistics
            print("\n📈 FINAL DATASET STATISTICS:")
            print(cleaned_data.describe().round(2))
            
            # Optional: Create visualizations
            print("\n🎨 To create visualizations, run:")
            print("# create_visualization(original_data, cleaned_data)")
        else:
            print("❌ Failed to process the dataset.")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {diabetes_file_path}")
        print("Please check the file path and ensure the file exists.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Alternative function for when you have the DataFrame already loaded - MOVED UP