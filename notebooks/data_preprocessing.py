from imports import *

def read_and_clean_data(file_path='../data/30_yr_stock_market_data.csv'):

    # Read the CSV data
    data = pd.read_csv(file_path)
    print("Initial Shape:", data.shape)
    
    # Calculate the percentage of NaN values in each column
    nan_percentage_per_column = (data.isna().sum() / len(data)) * 100
    print("NaN Percentage per Column:")
    print(nan_percentage_per_column)
    
    # Identify columns where the percentage of NaN values is greater than 25%
    columns_to_drop = nan_percentage_per_column[nan_percentage_per_column > 25].index
    # Drop these columns from the DataFrame
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Set time horizon to 20 years
    data['Date'] = pd.to_datetime(data['Date'])
    start_date = data['Date'].max() - pd.DateOffset(years=20)
    data = data[data['Date'] >= start_date]
    
    # Drop columns with all NaN values and rows with any NaN or blank values
    data = data.dropna(axis=1, how='all').dropna(axis=0, how='any')
    
    # Final percentage of NaN values in each column after cleaning
    final_nan_percentage_per_column = (data.isna().sum() / len(data)) * 100
    print("Final NaN Percentage per Column:")
    print(final_nan_percentage_per_column)
    
    return data
