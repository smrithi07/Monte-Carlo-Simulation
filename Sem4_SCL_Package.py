# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:47:52 2024

@author: dirav
"""

import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind
import seaborn as sns

sns.set_style('whitegrid')

def budget_for_sales_commissions():
    num_data_points = 365
    num_reps = 500
    num_simulations = 1000
    avg = 1
    std_dev = 0.1
    sales_target_values = [75_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    sales_target_prob = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]

    def calc_commission_rate(x):
        if x <= 0.90:
            return 0.02
        elif x <= 0.99:
            return 0.03
        else:
            return 0.04

    def calculate_additional_stats(df):
        stats = {
            'Average Sales': df['Sales'].mean(),
            'Min Sales': df['Sales'].min(),
            'Max Sales': df['Sales'].max(),
            'Median Commission Amount': df['Commission_Amount'].median(),
            'Total Sales': df['Sales'].sum(),
            'Total Commission Amount': df['Commission_Amount'].sum(),
            'Total Sales Target': df['Sales_Target'].sum(),
            '95% Confidence Interval (Sales)': t.interval(0.95, len(df['Sales']) - 1, loc=df['Sales'].mean(),
                                                          scale=df['Sales'].sem()),
            '95% Confidence Interval (Commission Amount)': t.interval(0.95, len(df['Commission_Amount']) - 1,
                                                                      loc=df['Commission_Amount'].mean(),
                                                                      scale=df['Commission_Amount'].sem())
        }
        return pd.Series(stats, dtype=object)

    def plot_data(df):
        plt.figure(figsize=(16, 6))

        plt.subplot(2, 3, 1)
        sns.kdeplot(df['Sales'], fill=True)
        plt.title('Density Plot of Sales')

        plt.subplot(2, 3, 2)
        sns.scatterplot(data=df, x='Sales', y='Commission_Amount')
        plt.title('Scatter Plot of Sales vs Commission Amount')

        plt.subplot(2, 3, 3)
        sns.histplot(df['Sales'], kde=True)
        plt.title('Histogram of Sales')

        plt.subplot(2, 3, 4)
        sns.boxplot(data=df, y='Sales')
        plt.title('Boxplot of Sales')

        plt.subplot(2, 3, 5)
        sns.boxplot(data=df, y='Commission_Amount')
        plt.title('Boxplot of Commission Amount')

        plt.subplot(2, 3, 6)
        sns.scatterplot(data=df, x='Sales_Target', y='Sales')
        plt.title('Scatter Plot of Sales vs Sales Target')

        plt.tight_layout()
        plt.show()

    simulation_results = []
    for i in range(num_simulations):
        sales_target = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)
        pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)

        df = pd.DataFrame(index=range(num_reps), data={'Pct_To_Target': pct_to_target,
                                                       'Sales_Target': sales_target})

        df['Sales'] = df['Pct_To_Target'] * df['Sales_Target']

        df['Commission_Rate'] = df['Pct_To_Target'].apply(calc_commission_rate)
        df['Commission_Amount'] = df['Commission_Rate'] * df['Sales']

        simulation_results.append([df['Sales'].sum().round(0),
                                   df['Commission_Amount'].sum().round(0),
                                   df['Sales_Target'].sum().round(0)])

    results_df = pd.DataFrame.from_records(simulation_results, columns=['Sales',
                                                                        'Commission_Amount',
                                                                        'Sales_Target'])

    additional_stats = calculate_additional_stats(results_df)
    print(additional_stats)

    plot_data(results_df)

    return results_df

def perform_sensitivity_analysis(df, variable, values):
    """
    Perform sensitivity analysis by changing a variable in the DataFrame and calculating
    the resulting Total Commission Amount for each value in the 'values' list.

    Parameters:
        df (DataFrame): DataFrame containing the sales data.
        variable (str): Variable name to change.
        values (list): List of values to test for the variable.

    Returns:
        list: List of Total Commission Amounts for each value.
    """
    results = []
    for value in values:
        df_copy = df.copy()
        df_copy[variable] = value
        stats = calculate_additional_stats(df_copy)
        results.append(stats['Total Commission Amount'])
    return results

def birthday_problem():
    def generate_birthdays(num_people):
        birthdays = []
        for _ in range(num_people):
            birthday = random.randint(1, 365)
            birthdays.append(birthday)
        return birthdays

    def has_duplicates(birthdays):
        unique_birthdays = set()
        for birthday in birthdays:
            if birthday in unique_birthdays:
                return True
            unique_birthdays.add(birthday)
        return False

    def birthday_simulation(num_people, num_simulations):
        count_duplicates = 0
        probabilities = []
        for _ in range(num_simulations):
            birthdays = generate_birthdays(num_people)
            if has_duplicates(birthdays):
                count_duplicates += 1
            probability = count_duplicates / (_ + 1)
            probabilities.append(probability)
        return probabilities

    def plot_probabilities(probabilities):
        plt.plot(range(1, len(probabilities) + 1), probabilities)
        plt.xlabel('Number of Simulations')
        plt.ylabel('Probability')
        plt.title('Probability of Shared Birthday in a Group')
        plt.grid(True)
        plt.show()

    num_people = 23
    num_simulations = 1000
    probabilities = birthday_simulation(num_people, num_simulations)
    plot_probabilities(probabilities)
    final_probability = probabilities[-1]
    print(f"Final Probability of at least two people sharing a birthday in a group of {num_people}: {final_probability:.4f}")

def monty_hall_problem():
    def monty_hall_simulation(num_simulations, switch=False):
        wins = 0
        for _ in range(num_simulations):
            car_door = random.randint(1, 3)
            contestant_choice = random.randint(1, 3)

            doors = [1, 2, 3]
            doors.remove(car_door)
            if contestant_choice in doors:
                doors.remove(contestant_choice)
            revealed_door = random.choice(doors)

            if switch:
                doors = [1, 2, 3]
                doors.remove(contestant_choice)
                doors.remove(revealed_door)
                contestant_choice = doors[0]

            if contestant_choice == car_door:
                wins += 1
        return wins

    def plot_results(switch_wins, stick_wins):
        categories = ['Switch Wins', 'Stick Wins']
        values = [switch_wins, stick_wins]
        plt.bar(categories, values, color=['blue', 'green'])
        plt.xlabel('Strategy')
        plt.ylabel('Number of Wins')
        plt.title('Monty Hall Simulation Results')
        plt.show()

    num_simulations = 10000
    switch_wins = monty_hall_simulation(num_simulations, switch=True)
    stick_wins = monty_hall_simulation(num_simulations, switch=False)
    switch_probability = switch_wins / num_simulations
    stick_probability = stick_wins / num_simulations

    print(f"Number of wins when switching: {int(switch_wins)}")
    print(f"Number of wins when sticking: {int(stick_wins)}")

    print(f"Winning probability when switching: {switch_probability:.4f}")
    print(f"Winning probability when sticking: {stick_probability:.4f}")

    plot_results(int(switch_wins), int(stick_wins))

def calculate_additional_stats(df):
    stats = {
        'Average Sales': df['Sales'].mean(),
        'Min Sales': df['Sales'].min(),
        'Max Sales': df['Sales'].max(),
        'Median Commission Amount': df['Commission_Amount'].median(),
        'Total Sales': df['Sales'].sum(),
        'Total Commission Amount': df['Commission_Amount'].sum(),
        'Total Sales Target': df['Sales_Target'].sum(),
        '95% Confidence Interval (Sales)': t.interval(0.95, len(df['Sales']) - 1, loc=df['Sales'].mean(),
                                                      scale=df['Sales'].sem()),
        '95% Confidence Interval (Commission Amount)': t.interval(0.95, len(df['Commission_Amount']) - 1,
                                                                  loc=df['Commission_Amount'].mean(),
                                                                  scale=df['Commission_Amount'].sem())
    }
    return pd.Series(stats, dtype=object)

def main():
    print("\n")
    print("******************************************")
    print("*   Scientific Computing Lab Package     *")
    print("*      Monte Carlo Simulation            *")
    print("******************************************")
    print("*                                        *")
    print("*        Done by: Diravina - 22PD12      *")
    print("*                Smrithi - 22PD33        *")
    print("*                                        *")
    print("******************************************")
    while True:
        print("******************************************")
        print("*                                        *")
        print("* Choose your application problem:       *")
        print("*                                        *")
        print("* 1. Budget for Sales Commissions         *")
        print("* 2. Birthday Problem                     *")
        print("* 3. Monty Hall Problem                   *")
        print("* 4. Exit                                *")
        print("*                                        *")
        print("******************************************")
        choice = input("\nEnter your choice: ")

        if choice == '1':
            print("\nBudget for Sales Commissions:")
            df = budget_for_sales_commissions()
            variable = input("Enter the variable for sensitivity analysis (Sales_Target, Commission_Rate, etc.): ")
            values_str = input("Enter the values to test (comma-separated): ")
            values = [float(val.strip()) for val in values_str.split(',')]
            results = perform_sensitivity_analysis(df, variable, values)
            print("Sensitivity Analysis Results:")
            for val, result in zip(values, results):
                print(f"{variable} = {val}: Total Commission Amount = {result:.2f}")
        elif choice == '2':
            print("\nBirthday Problem:")
            birthday_problem()
        elif choice == '3':
            print("\nMonty Hall Problem:")
            monty_hall_problem()
        elif choice == '4':
            print("\nExiting the program...")
            print("Thank You")
            break
        else:
            print("\nInvalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
