#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def laad_data(bestand):
    """
    Laad de CSV-data in een pandas DataFrame.
    """
    return pd.read_csv(bestand)

def plot_individuele_metrics(df, label="Algoritme"):
    """
    Plot de 6 metrics (TotalReward, ApplesEaten, AvgLoss, Epsilon, StepsPerApple, TotalTime)
    over de episodes in een subplot-overzicht.
    """
    episodes = df['Episode']
    
    plt.figure(figsize=(15,10))
    
    # Total Reward per Episode
    plt.subplot(2,3,1)
    plt.plot(episodes, df['TotalReward'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Apples Eaten per Episode
    plt.subplot(2,3,2)
    plt.plot(episodes, df['ApplesEaten'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Apples Eaten")
    plt.title("Apples Eaten per Episode")
    plt.legend()

    # Average Loss per Episode
    plt.subplot(2,3,3)
    plt.plot(episodes, df['AvgLoss'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Avg Loss")
    plt.title("Average Loss per Episode")
    plt.legend()

    # Epsilon per Episode
    plt.subplot(2,3,4)
    plt.plot(episodes, df['Epsilon'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon per Episode")
    plt.legend()

    # Steps Per Apple per Episode
    plt.subplot(2,3,5)
    plt.plot(episodes, df['StepsPerApple'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Steps Per Apple")
    plt.title("Steps Per Apple per Episode")
    plt.legend()

    # Total Time per Episode
    plt.subplot(2,3,6)
    plt.plot(episodes, df['TotalTime'], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Time")
    plt.title("Total Time per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_vergelijking(df1, df2, label1="Algoritme 1", label2="Algoritme 2"):
    """
    Vergelijk de metrics van twee algoritmes door ze in dezelfde grafieken weer te geven.
    """
    metrics = ['TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime']
    titels = ["Total Reward", "Apples Eaten", "Average Loss", "Epsilon", "Steps Per Apple", "Total Time"]

    plt.figure(figsize=(15,10))
    
    for i, (metric, titel) in enumerate(zip(metrics, titels), start=1):
        plt.subplot(2,3,i)
        plt.plot(df1['Episode'], df1[metric], label=label1)
        plt.plot(df2['Episode'], df2[metric], label=label2)
        plt.xlabel("Episode")
        plt.ylabel(titel)
        plt.title(f"{titel} per Episode")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_histogrammen(df, label="Algoritme"):
    """
    Plot histogrammen voor elke metric zodat je de verdeling kunt zien.
    """
    metrics = ['TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime']
    
    plt.figure(figsize=(15,10))
    for i, metric in enumerate(metrics, start=1):
        plt.subplot(2,3,i)
        plt.hist(df[metric].dropna(), bins=30, alpha=0.7, label=label)
        plt.xlabel(metric)
        plt.title(f"Histogram van {metric}")
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlatie_heatmap(df, label="Algoritme"):
    """
    Plot een heatmap van de correlaties tussen de metrics.
    """
    plt.figure(figsize=(8,6))
    # We gebruiken hier alleen de numerieke kolommen met de metrics
    correlatie = df[['TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime']].corr()
    sns.heatmap(correlatie, annot=True, cmap='coolwarm')
    plt.title(f"Correlatie Heatmap voor {label}")
    plt.show()

def main():
    # parser = argparse.ArgumentParser(description="Visualiseer de resultaten van een DQN-algoritme")
    # parser.add_argument("bestand1", help="./results.csv")
    # parser.add_argument("--bestand2", help="../Q-Learning/results.csv", default=None)
    # args = parser.parse_args()
    
    # Laad de data
    df1 = laad_data("./results.csv")
    
    # Als een tweede bestand is meegegeven, maak een vergelijking
    # if args.bestand2:
    df2 = laad_data("../Q-Learning/results.csv")
    plot_vergelijking(df1, df2)
    # else:
    #     plot_individuele_metrics(df1)
    
    # Extra visualisaties: histogrammen en correlatie heatmap
    plot_histogrammen(df1)
    plot_correlatie_heatmap(df1)

if __name__ == '__main__':
    main()
