import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
import pandas as pd
import statsmodels.api as sm


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