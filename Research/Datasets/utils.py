import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_stats(df, name):
    size_bins = [0, 0.1, 0.25, 0.5, 1]
    size_labels = ['малий', 'середній', 'великий', 'дуже великий']
    df['drone_size_category'] = pd.cut(df['scaled_drone_size'], bins=size_bins, labels=size_labels)

    aspect_bins = [0, 0.5, 1, 1.5, 2, float('inf')]
    aspect_labels = ['вузький', 'високий', 'квадратний', 'широкий', 'дуже широкий']
    df['drone_aspect_ratio_category'] = pd.cut(df['drone_aspect_ratio'], bins=aspect_bins, labels=aspect_labels)

    size_counts = df['drone_size_category'].value_counts().sort_index()

    aspect_ratio_counts = df['drone_aspect_ratio_category'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(df["drone_center_x"], df["drone_center_y"], bins=50, cmap='magma')
    plt.colorbar(h[3], label='Щільність')
    #ax.set_title('Теплова мапа розташування дронів на зображенні')
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    ax.set_xlabel('Позиція X')
    ax.set_ylabel('Позиція Y')
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["drone_center_x"], df["drone_center_y"], c='blue', label="Розташування дрона", alpha=0.6)
    #ax.set_title('Розподіл розташування дронів на зображенні')
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    ax.set_xlabel('Позиція X')
    ax.set_ylabel('Позиція Y')
    ax.legend()
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    size_counts.plot(kind='bar', color='green', alpha=0.7, ax=ax)
    #ax.set_title('Кількість дронів за категорією розміру')
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    ax.set_xlabel('Категорія розміру дрону')
    ax.set_ylabel('Кількість')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    aspect_ratio_counts.plot(kind='bar', color='purple', alpha=0.7, ax=ax)
    #ax.set_title('Кількість дронів за категорією співвідношення сторін')
    ax.set_xlabel('Категорія співвідношення сторін дрону')
    ax.set_ylabel('Кількість')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True)
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["scaled_drone_size"], bins=np.arange(0, 1, 0.01), color='blue', alpha=0.7)
    #ax.set_title('Розподіл розмірів дронів відносно розміру зображення')
    ax.set_xlabel('Розмір дрону')
    ax.set_ylabel('Кількість')
    ax.grid(True)
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["drone_aspect_ratio"], bins=np.arange(0, 3, 0.01), color='purple', alpha=0.7)
    #ax.set_title('Розподіл співвідношень сторін дронів')
    ax.set_xlabel('Співвідношення сторін дрону')
    ax.set_ylabel('Кількість')
    ax.grid(True)
    fig.text(0.5, 0.01, name, ha='center', fontsize=13, alpha=1)
    plt.show()
