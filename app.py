# Streamlit app for simulations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import seaborn as sns
import random

sns.set_style('whitegrid')


def budget_for_sales_commissions():
    st.title("Budget for Sales Commissions")
    num_reps = 500
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

    # Simulation
    sales_target = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)
    pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)

    df = pd.DataFrame(index=range(num_reps), data={
        'Pct_To_Target': pct_to_target,
        'Sales_Target': sales_target
    })

    df['Sales'] = df['Pct_To_Target'] * df['Sales_Target']
    df['Commission_Rate'] = df['Pct_To_Target'].apply(calc_commission_rate)
    df['Commission_Amount'] = df['Commission_Rate'] * df['Sales']

    st.write("### Summary of Sales Data")
    st.dataframe(df.head())

    # Plotting
    st.write("### Visualizations")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.kdeplot(df['Sales'], ax=axs[0], fill=True)
    axs[0].set_title("Density Plot of Sales")
    sns.scatterplot(data=df, x='Sales', y='Commission_Amount', ax=axs[1])
    axs[1].set_title("Scatter Plot of Sales vs Commission")
    st.pyplot(fig)

    st.write("### Statistics")
    stats = {
        'Average Sales': df['Sales'].mean(),
        'Total Sales': df['Sales'].sum(),
        'Total Commission Amount': df['Commission_Amount'].sum()
    }
    st.json(stats)


def birthday_problem():
    st.title("Birthday Problem")

    def generate_birthdays(num_people):
        return [random.randint(1, 365) for _ in range(num_people)]

    def has_duplicates(birthdays):
        return len(set(birthdays)) != len(birthdays)

    num_people = st.slider("Select the number of people in the group:", 1, 100, 23)
    num_simulations = st.slider("Select the number of simulations:", 100, 10000, 1000)

    probabilities = []
    count_duplicates = 0
    for _ in range(num_simulations):
        birthdays = generate_birthdays(num_people)
        if has_duplicates(birthdays):
            count_duplicates += 1
        probabilities.append(count_duplicates / (_ + 1))

    # Plotting
    st.write("### Probability Plot")
    plt.plot(range(1, len(probabilities) + 1), probabilities)
    plt.xlabel("Number of Simulations")
    plt.ylabel("Probability")
    plt.title("Probability of Shared Birthday in a Group")
    st.pyplot(plt)

    st.write(
        f"The final probability of at least two people sharing a birthday is **{probabilities[-1]:.4f}**."
    )


def monty_hall_problem():
    st.title("Monty Hall Problem")

    num_simulations = st.slider("Select the number of simulations:", 100, 10000, 1000)
    switch = st.radio("Do you want to switch doors?", ("Yes", "No"))

    def monty_hall_simulation(num_simulations, switch):
        wins = 0
        for _ in range(num_simulations):
            car_door = random.randint(1, 3)
            contestant_choice = random.randint(1, 3)
            doors = [1, 2, 3]
            doors.remove(car_door)
            if contestant_choice in doors:
                doors.remove(contestant_choice)
            revealed_door = doors[0]

            if switch == "Yes":
                remaining_door = [d for d in [1, 2, 3] if d != contestant_choice and d != revealed_door][0]
                contestant_choice = remaining_door

            if contestant_choice == car_door:
                wins += 1
        return wins

    wins = monty_hall_simulation(num_simulations, switch)
    st.write(
        f"### Number of Wins: **{wins}** out of **{num_simulations}** simulations."
    )
    st.write(
        f"Winning probability: **{(wins / num_simulations):.4f}**."
    )


# Main Streamlit App
st.sidebar.title("Monte Carlo Simulations")
options = ["Budget for Sales Commissions", "Birthday Problem", "Monty Hall Problem"]
choice = st.sidebar.radio("Choose a simulation:", options)

if choice == "Budget for Sales Commissions":
    budget_for_sales_commissions()
elif choice == "Birthday Problem":
    birthday_problem()
elif choice == "Monty Hall Problem":
    monty_hall_problem()
