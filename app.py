import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Veri kümesini oku
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv")
df.dropna(subset=['Price'], inplace=True)

# Eksik değerleri doldur
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

for col in numerical_cols:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

# Dummy değişkenleri oluştur
potential_categorical_cols = ['Type', 'AirBags', 'DriveTrain', 'Origin']
df = pd.get_dummies(df, columns=potential_categorical_cols, drop_first=True)

# Model için veriler
X = df[['EngineSize', 'Horsepower', 'Weight', 'Length'] + [col for col in df.columns if any(cat in col for cat in potential_categorical_cols)]]
y = df['Price']

# Modeli eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Model ve kolon isimlerini kaydet
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
print("Model ve kolonlar kaydedildi.")


import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

# Modeli ve kolon isimlerini yükle
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_features.pkl')

# Kategorik seçenekler
type_options = ['Small', 'Sporty', 'Compact', 'Large', 'Midsize', 'Van', 'Wagon']
airbags_options = ['None', 'Driver only', 'Driver & Passenger']
drivetrain_options = ['Front', 'Rear', '4WD']
origin_options = ['USA', 'non-USA']

def predict_price():
    try:
        # Sayısal girişleri al
        engine_size = float(entry_engine.get())
        horsepower = float(entry_hp.get())
        weight = float(entry_weight.get())
        length = float(entry_length.get())

        # Kategorik girişleri al
        type_val = combo_type.get()
        airbags_val = combo_airbags.get()
        drivetrain_val = combo_drivetrain.get()
        origin_val = combo_origin.get()

        # DataFrame oluştur
        input_data = {
            'EngineSize': [engine_size],
            'Horsepower': [horsepower],
            'Weight': [weight],
            'Length': [length],
            f'Type_{type_val}': [1],
            f'AirBags_{airbags_val}': [1],
            f'DriveTrain_{drivetrain_val}': [1],
            f'Origin_{origin_val}': [1]
        }

        # Eksik dummy'leri 0 ile doldur
        for col in model_columns:
            if col not in input_data:
                input_data[col] = [0]

        # Doğru sıraya sok
        input_df = pd.DataFrame(input_data)[model_columns]

        # Tahmini yap
        price = model.predict(input_df)[0]
        messagebox.showinfo("Tahmin Sonucu", f"Tahmini Fiyat: {price:.2f} $")

    except Exception as e:
        messagebox.showerror("Hata", f"Hata:\n{e}")

# Arayüz
root = tk.Tk()
root.title("Araç Fiyat Tahmini (Gelişmiş)")
root.geometry("400x500")

tk.Label(root, text="Sayısal Özellikler", font=("Arial", 14, "bold")).pack(pady=10)

tk.Label(root, text="Engine Size").pack()
entry_engine = tk.Entry(root)
entry_engine.pack()

tk.Label(root, text="Horsepower").pack()
entry_hp = tk.Entry(root)
entry_hp.pack()

tk.Label(root, text="Weight").pack()
entry_weight = tk.Entry(root)
entry_weight.pack()

tk.Label(root, text="Length").pack()
entry_length = tk.Entry(root)
entry_length.pack()

tk.Label(root, text="Kategorik Özellikler", font=("Arial", 14, "bold")).pack(pady=10)

tk.Label(root, text="Type").pack()
combo_type = ttk.Combobox(root, values=type_options)
combo_type.pack()

tk.Label(root, text="AirBags").pack()
combo_airbags = ttk.Combobox(root, values=airbags_options)
combo_airbags.pack()

tk.Label(root, text="DriveTrain").pack()
combo_drivetrain = ttk.Combobox(root, values=drivetrain_options)
combo_drivetrain.pack()

tk.Label(root, text="Origin").pack()
combo_origin = ttk.Combobox(root, values=origin_options)
combo_origin.pack()

tk.Button(root, text="Tahmin Et", command=predict_price, bg="green", fg="white").pack(pady=20)

root.mainloop()
