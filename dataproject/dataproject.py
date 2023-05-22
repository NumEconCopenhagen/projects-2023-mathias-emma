import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import statsmodels.api as sm
plt.style.use('ggplot')
import geopandas as gpd
from fuzzywuzzy import process


def plot_time(df):
    subset_df = df[df['mun'] == 'All Denmark']

    # Convert 'year' to numeric data type
    subset_df['year'] = pd.to_numeric(subset_df['year'])

    fig, ax1 = plt.subplots()

    # Plot 'exp' on the first axis
    ax1.plot(subset_df['year'], subset_df['exp_per_cap'], color='b')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Exp', color='b')

    # Create a twin axis
    ax2 = ax1.twinx()

    # Plot 'loan' on the second axis
    ax2.plot(subset_df['year'], subset_df['loan_per_cap'], color='r')
    ax2.set_ylabel('Loan', color='r')

    # Set tick locations and labels for x-axis
    xticks = np.arange(subset_df['year'].min(), subset_df['year'].max()+1, 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)

    plt.title('Plot of Exp and Loan over Time')
    plt.show()

def plot_scatter(df_2):
    subset_df = df_2 #[df['year'] == 'YYYY']

    # Create a scatter plot with 'exp' on the first axis and 'loan' on the second axis
    ax = sns.regplot(x='exp_per_cap', y='loan_per_cap', data=subset_df)

    # Set plot title and labels
    plt.title('Scatter Plot of Exp and Loan from 2009-2022')
    plt.xlabel('Exp pr. capita')
    plt.ylabel('Loan pr. capita')

    # Add a regression equation
    slope, intercept = np.polyfit(subset_df['exp_per_cap'], subset_df['loan_per_cap'], 1)
    eq = f'y = {slope:.2f}x + {intercept:.2f}'
    ax.annotate(eq, xy=(0.05, 0.95), xycoords='axes fraction')

    # Display the plot
    plt.show()

    subset_df = df_2 #[df['year'] == 'YYYY']

    # Create a regression model with 'exp_per_cap' and 'Loan_per_cap'
    X = subset_df[['exp_per_cap']]
    y = subset_df['loan_per_cap']
    X = sm.add_constant(X)
    model1 = sm.OLS(y, X).fit()
    print(model1.summary())

def plot_year(df_2, year):
    # Create a figure
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the first subplot
    subset_df1 = df_2[df_2['year'] == year]
    sns.regplot(x='exp_per_cap', y='loan_per_cap', data=subset_df1, ax=ax1)
    slope, intercept = np.polyfit(subset_df1['exp_per_cap'], subset_df1['loan_per_cap'], 1)
    eq = f'y = {slope:.2f}x + {intercept:.2f}'
    ax1.annotate(eq, xy=(0.05, 0.95), xycoords='axes fraction')
    ax1.set_title('Scatter Plot of Exp and Loan in 2018')
    ax1.set_xlabel('Exp pr. capita')
    ax1.set_ylabel('Loan pr. capita')

    # adjust the layout and display the plot
    fig.tight_layout()
    plt.show()


def map_plot(denmark_map,df_mun,year):
    # Load municipality boundaries from a shapefile
    denmark_map = gpd.read_file('Map_data/DNK_adm2.shp')
    
    # Rename the 'NAME_2' column to 'mun'
    denmark_map = denmark_map.rename(columns={'NAME_2': 'mun'})

    # Define a function to match strings based on similarity score
    def match_strings(x, choices):
        best_match = process.extractOne(x, choices)
        return best_match[0] if best_match[1] >= 80 else None

    def prepare_data(df_mun, year, denmark_map):
        # Load the expenditure data into a pandas dataframe
        df_2_mun = df_mun[df_mun['year'] == str(year)]

        # Merge the two dataframes based on similar 'mun' names
        df_2_mun['mun_match'] = df_2_mun['mun'].apply(lambda x: match_strings(x, denmark_map['mun']))
        df_2_mun = df_2_mun.dropna(subset=['mun_match'])
        denmark_map_year = denmark_map.merge(df_2_mun, left_on='mun', right_on='mun_match')

        # Set the color scale based on the expenditure values
        exp_min = denmark_map_year['exp'].min()
        exp_max = denmark_map_year['exp'].max()
        denmark_map_year['color'] = (denmark_map_year['exp'] - exp_min) / (exp_max - exp_min)

        return denmark_map_year, exp_min, exp_max

    denmark_map_2012, exp_min_2012, exp_max_2012 = prepare_data(df_mun, 2012, denmark_map)
    denmark_map_2018, exp_min_2018, exp_max_2018 = prepare_data(df_mun, 2018, denmark_map)

    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Set the color bar range based on min and max values from both years
    exp_min = min(exp_min_2012, exp_min_2018)
    exp_max = max(exp_max_2012, exp_max_2018)

    # Plot the maps
    for ax, denmark_map_year, year in zip(axs, [denmark_map_2012, denmark_map_2018], [2012, 2018]):
        denmark_map_year.plot(column='color', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='black')
        ax.set_title(f'Year {year}')
        ax.set_axis_off()

    cbar_ax = fig.add_axes([0.48, 0.2, 0.01, 0.7])

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=exp_min, vmax=exp_max))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel('Expenditure on material')


    # Add main title
    plt.suptitle('Analysis of Material Expenditure by Municipality in Denmark', fontsize=24)
    plt.subplots_adjust(top=0.95)

    plt.show()