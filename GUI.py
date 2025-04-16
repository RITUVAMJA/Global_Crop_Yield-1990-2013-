import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Load model and scaler
best_rf_model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Manual category mappings (match these with training data)
area_map = {
    'India': 0,
    'Brazil': 1,
    'Mexico': 2,
    'Pakistan': 3,
    'China': 4,
    'Other_Area': 5
}

item_map = {
    'Potatoes': 0,
    'Wheat': 1,
    'Rice': 2,
    'Maize': 3,
    'Barley': 4,
    'Other_Item': 5
}

temp_map = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
rainfall_map = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}

# GUI setup
app = tk.Tk()
app.title("🌿 Crop Yield Predictor")
app.geometry("400x600")
app.resizable(False, False)

tk.Label(app, text="Area (Country):").pack()
area_var = tk.StringVar()
ttk.Combobox(app, textvariable=area_var, values=list(area_map.keys())).pack()

tk.Label(app, text="Crop (Item):").pack()
item_var = tk.StringVar()
ttk.Combobox(app, textvariable=item_var, values=list(item_map.keys())).pack()

tk.Label(app, text="Year:").pack()
year_entry = tk.Entry(app)
year_entry.pack()

tk.Label(app, text="Rainfall Category:").pack()
rainfall_var = tk.StringVar()
ttk.Combobox(app, textvariable=rainfall_var, values=list(rainfall_map.keys())).pack()

tk.Label(app, text="Temperature Category:").pack()
temp_var = tk.StringVar()
ttk.Combobox(app, textvariable=temp_var, values=list(temp_map.keys())).pack()

tk.Label(app, text="Pesticides (Tonnes):").pack()
pesticides_entry = tk.Entry(app)
pesticides_entry.pack()

result_label = tk.Label(app, text="Prediction will appear here", font=("Arial", 12), pady=10)
result_label.pack()

# Prediction logic
def predict_yield():
    try:
        area = area_var.get()
        item = item_var.get()
        year = int(year_entry.get())
        rainfall = rainfall_var.get()
        temp = temp_var.get()
        pesticides = float(pesticides_entry.get())

        # Manual encoding
        area_encoded = area_map[area]
        item_encoded = item_map[item]
        rainfall_encoded = rainfall_map[rainfall]
        temp_encoded = temp_map[temp]

        # Standardize year and pesticides
        dummy = np.zeros((1, 3))  # [pesticides, year, yield]
        dummy[0, 0] = pesticides
        dummy[0, 1] = year
        dummy[0, 2] = 0
        scaled = scaler.transform(dummy)
        scaled_pesticides = scaled[0, 0]
        scaled_year = scaled[0, 1]

        input_data = np.array([[area_encoded, item_encoded, scaled_year, rainfall_encoded, scaled_pesticides, temp_encoded]])
        scaled_prediction = best_rf_model.predict(input_data)[0]

        dummy[0, 2] = scaled_prediction
        original_yield = scaler.inverse_transform(dummy)[0, 2]

        result_label.config(text=f"🌾 Predicted Yield: {original_yield:.2f} hg/ha")

    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(app, text="Predict Yield", command=predict_yield, bg="green", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()
