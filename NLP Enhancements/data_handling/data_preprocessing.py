import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import logging
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os


# -------------------------------------------------------------------------------
# added this script because here a best use case of the data handling and that is it also stores the images and graphs , charts in one file and also generate and combined GUI interface where we can see the whole records 
# -------------------------------------------------------------------------------


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.scaler = None

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def clean_text(self, text):
        """
        Clean text data by removing special characters, numbers, and converting to lowercase.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        return text

    def remove_stopwords(self, text):
        """
        Remove stopwords from the text.
        """
        tokens = word_tokenize(text)
        return ' '.join([token for token in tokens if token not in self.stop_words])

    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataframe.
        """
        if strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        else:
            return df.dropna()

    def scale_features(self, df, method='standard'):
        """
        Scale numerical features.
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            logging.warning("Invalid scaling method. Using StandardScaler.")
            self.scaler = StandardScaler()

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

    def perform_eda(self, df):
        """
        Perform basic Exploratory Data Analysis (EDA) and create visualizations.
        """
        logging.info("Performing Exploratory Data Analysis")
        
        # Create directories for saving plots
        eda_dir = os.path.join('NLP Enhancements', 'data_handling', 'eda_plots')
        os.makedirs(eda_dir, exist_ok=True)
        
        # Display basic information about the dataset
        info_str = df.info(memory_usage='deep')
        desc_str = df.describe().to_string()
        
        # Plot histograms for numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(nrows=(len(numeric_columns)+1)//2, ncols=2, figsize=(15, 5*((len(numeric_columns)+1)//2)))
        for idx, col in enumerate(numeric_columns):
            row = idx // 2
            col_idx = idx % 2
            df[col].hist(ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Histogram of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, 'histograms.png'))
        plt.close()

        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'))
        plt.close()

        logging.info(f"EDA completed. Plots saved in '{eda_dir}' directory.")
        
        return info_str, desc_str

    def preprocess_data(self, df, text_column=None):
        """
        Preprocess the data by applying various cleaning and transformation steps.
        """
        logging.info("Starting data preprocessing")

        # Handle missing values
        df = self.handle_missing_values(df)

        # Scale numerical features
        df = self.scale_features(df)

        # Clean text data if a text column is specified
        if text_column and text_column in df.columns:
            df[text_column] = df[text_column].apply(self.clean_text)
            df[text_column] = df[text_column].apply(self.remove_stopwords)

        logging.info("Data preprocessing completed")
        return df

class DataVisualizationGUI:
    def __init__(self, master, df, info_str, desc_str):
        self.master = master
        self.df = df
        self.info_str = info_str
        self.desc_str = desc_str
        
        self.master.title("Data Visualization GUI")
        self.master.geometry("800x600")
        
        self.create_widgets()

    def create_widgets(self):
        # Create tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Info tab
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="Info")
        info_text = tk.Text(info_frame, wrap=tk.WORD)
        info_text.insert(tk.END, str(self.info_str))
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # Description tab
        desc_frame = ttk.Frame(self.notebook)
        self.notebook.add(desc_frame, text="Description")
        desc_text = tk.Text(desc_frame, wrap=tk.WORD)
        desc_text.insert(tk.END, self.desc_str)
        desc_text.pack(fill=tk.BOTH, expand=True)
        
        # Histogram tab
        hist_frame = ttk.Frame(self.notebook)
        self.notebook.add(hist_frame, text="Histograms")
        self.hist_canvas = tk.Canvas(hist_frame)
        self.hist_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Heatmap tab
        heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(heatmap_frame, text="Correlation Heatmap")
        self.heatmap_canvas = tk.Canvas(heatmap_frame)
        self.heatmap_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Load and display plots
        self.load_plots()

        # Bind resize event
        self.master.bind("<Configure>", self.on_resize)

    def load_plots(self):
        eda_dir = os.path.join('NLP Enhancements', 'data_handling', 'eda_plots')
        
        # Load histogram
        self.hist_img = Image.open(os.path.join(eda_dir, 'histograms.png'))
        self.hist_photo = ImageTk.PhotoImage(self.hist_img)
        self.hist_canvas.create_image(0, 0, anchor=tk.NW, image=self.hist_photo)
        
        # Load heatmap
        self.heatmap_img = Image.open(os.path.join(eda_dir, 'correlation_heatmap.png'))
        self.heatmap_photo = ImageTk.PhotoImage(self.heatmap_img)
        self.heatmap_canvas.create_image(0, 0, anchor=tk.NW, image=self.heatmap_photo)

    def on_resize(self, event):
        # Resize histogram
        width = self.hist_canvas.winfo_width()
        height = self.hist_canvas.winfo_height()
        resized_hist = self.hist_img.resize((width, height), Image.Resampling.LANCZOS)
        self.hist_photo = ImageTk.PhotoImage(resized_hist)
        self.hist_canvas.delete("all")
        self.hist_canvas.create_image(0, 0, anchor=tk.NW, image=self.hist_photo)

        # Resize heatmap
        width = self.heatmap_canvas.winfo_width()
        height = self.heatmap_canvas.winfo_height()
        resized_heatmap = self.heatmap_img.resize((width, height), Image.Resampling.LANCZOS)
        self.heatmap_photo = ImageTk.PhotoImage(resized_heatmap)
        self.heatmap_canvas.delete("all")
        self.heatmap_canvas.create_image(0, 0, anchor=tk.NW, image=self.heatmap_photo)

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data('NLP Enhancements/data_handling/Avalache_data.csv')
    
    if data is not None:
        # Preprocess data
        preprocessed_data = preprocessor.preprocess_data(data)
        
        # Perform EDA and get info and description strings
        info_str, desc_str = preprocessor.perform_eda(preprocessed_data)
        
        # Create and run GUI
        root = tk.Tk()
        app = DataVisualizationGUI(root, preprocessed_data, info_str, desc_str)
        root.mainloop()


