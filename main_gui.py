import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Import our custom modules
from src.preprocessing import load_and_prepare_data
from src.train_models import train_and_save_models, evaluate_model

# Path ke file CSV yang sudah ditentukan
CSV_FILE_PATH = "data/military_expenditure.csv"

def get_country_list():
    """Get the list of countries from the dataset."""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        return sorted(df["country"].dropna().unique())
    except FileNotFoundError:
        print(f"Warning: File not found at {CSV_FILE_PATH}")
        return []
    except Exception as e:
        print(f"Error loading country list: {e}")
        return []

def format_currency(value):
    """Format large numbers as currency with appropriate suffixes."""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f} Miliar"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f} Juta"
    elif value >= 1_000:
        return f"${value/1_000:.2f} Ribu"
    else:
        return f"${value:.2f}"

def run_prediction():
    """Run the prediction process and display results."""
    country = country_var.get()
    if not country or country == "Pilih negara...":
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Silakan pilih negara terlebih dahulu.")
        return
    
    try:
        # Show loading message
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Memproses data dan melatih model...\n")
        root.update()
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(country=country)
        
        # Train and save the models
        models = train_and_save_models(X_train, y_train)
        
        # Results dictionary to store predictions for each model
        results = {}
        
        # Load all saved models and make predictions
        for model_name, model in models.items():
            # Evaluate the model
            eval_results = evaluate_model(model, X_test, y_test)
            results[model_name] = {
                "metrics": eval_results,
                "predictions": eval_results["predictions"]
            }
        
        # Display results
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Hasil Prediksi untuk {country}:\n\n")
        
        # Get years from the test set
        years = X_test["year_original"].values if "year_original" in X_test.columns else X_test["year"].values
        
        # Show table of results
        result_text.insert(tk.END, "Perbandingan Nilai Aktual dan Prediksi (dalam USD):\n\n")
        result_text.insert(tk.END, f"{'Tahun':<10}{'Aktual':<20}{'Random Forest':<20}{'Linear':<20}{'Polynomial':<20}\n")
        result_text.insert(tk.END, "-" * 80 + "\n")
        
        for i, (year, actual) in enumerate(zip(years, y_test)):
            rf_pred = results["random_forest"]["predictions"][i]
            lr_pred = results["linear_regression"]["predictions"][i]
            poly_pred = results["polynomial"]["predictions"][i]
            
            result_text.insert(tk.END, f"{int(year):<10}{format_currency(actual):<20}{format_currency(rf_pred):<20}{format_currency(lr_pred):<20}{format_currency(poly_pred):<20}\n")
        
        result_text.insert(tk.END, "\n\nMetrik Evaluasi Model:\n\n")
        for model_name, result in results.items():
            metrics = result["metrics"]
            r2 = metrics["r2"]
            rmse = metrics["rmse"]
            
            model_display_name = {
                "random_forest": "Random Forest",
                "linear_regression": "Regresi Linear",
                "polynomial": "Regresi Polinomial"
            }.get(model_name, model_name)
            
            result_text.insert(tk.END, f"{model_display_name}:\n")
            result_text.insert(tk.END, f"  - R² Score: {r2:.4f} (semakin mendekati 1 semakin baik)\n")
            result_text.insert(tk.END, f"  - RMSE: {format_currency(rmse)} (semakin kecil semakin baik)\n\n")
        
        # Plot the results
        plot_predictions(country, years, y_test, results)
        
    except Exception as e:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Terjadi kesalahan:\n{str(e)}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()

def plot_predictions(country, years, y_test, results):
    """Plot predictions vs actual values."""
    # Clear previous plot if it exists
    if hasattr(plot_predictions, 'canvas') and plot_predictions.canvas is not None:
        plot_predictions.canvas.get_tk_widget().destroy()
    
    # Create larger figure
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)  # Perbesar ukuran kanvas

    # Sort years and corresponding values
    indices = np.argsort(years)
    years = years[indices]
    y_test = y_test[indices]
    
    # Plot actual values
    ax.plot(years, y_test, 'o-', label='Aktual', color='black', linewidth=2)
    
    # Plot predicted values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    model_names = {
        "random_forest": "Random Forest",
        "linear_regression": "Regresi Linear",
        "polynomial": "Regresi Polinomial"
    }
    
    for i, (model_name, result) in enumerate(results.items()):
        predictions = result["predictions"][indices]
        ax.plot(years, predictions, 'o--', label=model_names.get(model_name, model_name), 
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.7)
    
    # Add labels and title with larger font
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Pengeluaran Militer (USD)', fontsize=12)
    ax.set_title(f'Prediksi Pengeluaran Militer untuk {country}', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with larger font
    ax.legend(fontsize=10)
    
    # Rotate year labels and adjust padding
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis
    from matplotlib.ticker import FuncFormatter
    def billions_formatter(x, pos):
        return f'${x/1e9:.1f}B'
    ax.yaxis.set_major_formatter(FuncFormatter(billions_formatter))
    
    # Adjust layout with more padding
    plt.tight_layout(pad=5.0)  # Tambahkan padding untuk menghindari elemen terpotong
    
    # Create canvas with expanded space
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    
    # Get the widget and pack with more space
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Store the canvas reference
    plot_predictions.canvas = canvas

# GUI setup
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistem Prediksi Pengeluaran Militer")
    root.geometry("1000x900")
    
    # Kurangi padding pada header
    header_frame = tk.Frame(root, bg="#003366", padx=5, pady=0.5)  
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(
        header_frame, 
        text="Sistem Prediksi Pengeluaran Militer", 
        font=("Arial", 18, "bold"),
        fg="white",
        bg="#003366"
    )
    header.pack(pady=0.5)  # Kurangi padding
    
    subheader = tk.Label(
        header_frame,
        text="Analisis dan Prediksi Tren Pengeluaran Militer Negara",
        font=("Arial", 12),
        fg="white",
        bg="#003366"
    )
    subheader.pack(pady=0.5)  # Kurangi padding
    
    # Kurangi padding pada content frame
    content_frame = tk.Frame(root, padx=10, pady=0.5)  # Kurangi padding
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    # Kurangi padding pada informasi dataset
    info_frame = tk.Frame(content_frame)
    info_frame.pack(fill=tk.X, pady=0.5)  # Kurangi padding
    
    info_label = tk.Label(
        info_frame, 
        text=f"Dataset: military_expenditure.csv", 
        font=("Arial", 11, "italic"),
        fg="#555555"
    )
    info_label.pack(side=tk.LEFT, padx=0.5)  # Kurangi padding
    
    # Kurangi padding pada pemilihan negara
    country_frame = tk.Frame(content_frame)
    country_frame.pack(fill=tk.X, pady=0.5)  # Kurangi padding
    
    country_label = tk.Label(country_frame, text="Pilih Negara:", font=("Arial", 11))
    country_label.pack(side=tk.LEFT, padx=0.5)  # Kurangi padding
    
    country_var = tk.StringVar()
    countries = get_country_list()
    
    if not countries:
        country_menu = ttk.Combobox(country_frame, textvariable=country_var, state="readonly", width=40)
        country_menu.pack(side=tk.LEFT, padx=0.5)  # Kurangi padding
        country_menu.set("Data negara tidak tersedia")
        
        # Show error if file not found
        messagebox.showerror("Error", f"File dataset tidak ditemukan di {CSV_FILE_PATH}.\nPastikan file tersedia di lokasi yang benar.")
    else:
        country_menu = ttk.Combobox(country_frame, textvariable=country_var, values=countries, state="readonly", width=40)
        country_menu.pack(side=tk.LEFT, padx=0.5)  # Kurangi padding
        country_menu.set("Pilih negara...")
    
    # Kurangi padding pada tombol prediksi
    button_frame = tk.Frame(content_frame)
    button_frame.pack(fill=tk.X, pady=0.5)  # Kurangi padding
    
    predict_button = tk.Button(
        button_frame, 
        text="Analisis & Prediksi", 
        command=run_prediction, 
        bg="#55D4BA", 
        fg="white",
        font=("Arial", 12, "bold"),
        padx=10,  # Kurangi padding
        pady=1    # Kurangi padding
    )
    predict_button.pack()
    
    # Kurangi tinggi area hasil
    result_frame = tk.Frame(content_frame, relief=tk.GROOVE, bd=1)
    result_frame.pack(fill=tk.BOTH, expand=True, pady=0.5)  # Kurangi padding
    
    result_label = tk.Label(result_frame, text="Hasil Prediksi:", font=("Arial", 12, "bold"), anchor="w")
    result_label.pack(fill=tk.X, padx=10, pady=0.5)
    
    # Create scrollable text area for results
    result_scroll = tk.Scrollbar(result_frame)
    result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    result_text = tk.Text(result_frame, height=10, width=70, yscrollcommand=result_scroll.set)  # Kurangi tinggi
    result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=0.5)  # Kurangi padding
    result_scroll.config(command=result_text.yview)
    
    # Kurangi padding pada footer
    footer_frame = tk.Frame(root, bg="#f0f0f0", padx=5, pady=0.5)  # Kurangi padding
    footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    footer = tk.Label(
        footer_frame, 
        text="© 2025 Sistem Prediksi Pengeluaran Militer | Analisis Data Berbasis Machine Learning", 
        font=("Arial", 8),
        bg="#f0f0f0"
    )
    footer.pack(pady=0.5)  # Kurangi padding
    
    # Initialize the plot attribute
    plot_predictions.canvas = None
    
    # Display info about the dataset columns
    column_info = (
        "Kolom Dataset: country, iso3c, iso2c, year, Military expenditure (current USD), "
        "Military expenditure (% of general government expenditure), Military expenditure (% of GDP), "
        "adminregion, incomeLevel"
    )
    dataset_info = tk.Label(
        content_frame,
        text=column_info,
        font=("Arial", 9),
        fg="#666666", 
        wraplength=750,
        justify=tk.LEFT
    )
    dataset_info.pack(fill=tk.X, pady=(0, 10))
    
    root.mainloop()