from solids.imports import *
import pandas as pd
from dagster import op, Out
from sklearn.preprocessing import StandardScaler

# Feature Engineering Function
@op(
    out={
        "spread_data": Out(),
        "features": Out(),
    }
)
def create_spreads_and_more(df):
    # Function to label the spread differences inside the main function
    def label_spread(x):
        if x > 0.1:
            return 'divergence'  # Spread increased by more than 0.1
        elif x < -0.1:
            return 'convergence'  # Spread decreased by more than -0.1
        else:
            return 'steady'  # Spread change is within a smaller range

    # Drop redundant independent variables
    df = df.drop(
        ['Dow Jones (^DJI)', 'Nasdaq (^IXIC)', 'NYSE Composite (^NYA)', 'Russell 2000 (^RUT)',
         'Treasury Yield 5 Years (^FVX)', 'Treasury Bill 13 Week (^IRX)', 'Treasury Yield 30 Years (^TYX)'],
        axis=1
    )
    
    # Standardize the numerical data
    numeric_data = df.select_dtypes(include=[float, int])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns, index=df.index)

    # Calculate the spread between each index and the Treasury Yield 10 Years (^TNX)
    spread_data = pd.DataFrame(index=df_scaled.index)
    
    for column in df_scaled.columns[:-1]:  # Exclude the last column which is the Treasury Yield 10 Years
        spread_column_name = f"Spread_{column}_TNX"
        spread_data[spread_column_name] = df_scaled[column] - df_scaled['Treasury Yield 10 Years (^TNX)']

    # Ensure the index is a datetime object
    if not isinstance(spread_data.index, pd.DatetimeIndex):
        spread_data.index = pd.to_datetime(spread_data.index)
    
    # Add 'Year' as a feature
    spread_data['Year'] = spread_data.index.year
    year_column = spread_data['Year']
    spread_data.drop('Year', axis=1, inplace=True)
    spread_data.insert(0, 'Year', year_column)
    
    # Reorder columns to put 'Year' at the front
    spread_data = spread_data[['Year'] + [col for col in spread_data.columns if col != 'Year']]

    # Group by year and calculate the mean spread for each column
    average_spread_per_year = spread_data.groupby(spread_data['Year']).mean()

    # Create the new DataFrame with the differences
    df_diff = pd.DataFrame(index=spread_data.index)
    for column in spread_data.columns.difference(['Year']):
        # For each column, calculate the difference between the spread and the yearly average
        yearly_avg = average_spread_per_year[column].reindex(spread_data['Year']).values
        df_diff[column] = spread_data[column] - yearly_avg

    # Label the spread differences
    df_diff['spread_label'] = df_diff.apply(lambda row: label_spread(row.mean()), axis=1)

    return df_diff, df_scaled