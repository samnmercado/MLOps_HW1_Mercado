from solids.imports import *

# Function to label the spread differences
def label_spread(x):
    if x > 0.1:
        return 'divergence'  # Spread increased by more than 0.1
    elif x < -0.1:
        return 'convergence'  # Spread decreased by more than -0.1
    else:
        return 'steady'  # Spread change is within a smaller range

# Feature Engineering Function
@solid
def create_spreads_and_more(df):
    # Standardize the numerical data
    numeric_data = df.select_dtypes(include=[float, int])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns, index=df.index)

    # Drop redundant independent variables
    df_scaled = df_scaled.drop(
        ['Dow Jones (^DJI)', 'Nasdaq (^IXIC)', 'NYSE Composite (^NYA)', 'Russell 2000 (^RUT)',
         'Treasury Yield 5 Years (^FVX)', 'Treasury Bill 13 Week (^IRX)', 'Treasury Yield 30 Years (^TYX)'],
        axis=1
    )
    
    # Calculate the spread between each index and the Treasury Yield 10 Years (^TNX)
    spread_data = pd.DataFrame(index=df_scaled.index)
    
    for column in df_scaled.columns[:-1]:  # Exclude the last column which is the Treasury Yield 10 Years
        spread_column_name = f"Spread_{column}_TNX"
        spread_data[spread_column_name] = df_scaled[column] - df_scaled['Treasury Yield 10 Years (^TNX)']
    
    # Add 'Year' as a feature
    spread_data['Year'] = df_scaled.index.year
    
    # Reorder columns to put 'Year' at the front
    spread_data = spread_data[['Year'] + [col for col in spread_data.columns if col != 'Year']]

    # Group by year and calculate the mean spread for each column
    average_spread_per_year = spread_data.groupby(spread_data.index.year).mean()

    # Create the new DataFrame with the differences
    df_diff = pd.DataFrame(index=spread_data.index)
    for column in spread_data.columns.difference(['Year']):
        # For each column, calculate the difference between the spread and the yearly average
        yearly_avg = average_spread_per_year[column].reindex(spread_data['Year']).values
        df_diff[column] = spread_data[column] - yearly_avg

    # Apply the label_spread function to the differences
    df_labels = df_diff.applymap(label_spread)

    # Drop the first row
    df_labels_mapped = df_labels.iloc[1:]

    # Apply the mapping to convert labels from text to numbers
    mapping = {'divergence': 1, 'convergence': -1, 'steady': 0}
    df_labels_mapped = df_labels.applymap(lambda x: mapping.get(x, x))

    return spread_data, df_labels_mapped
