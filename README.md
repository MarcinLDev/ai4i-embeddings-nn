# AI4I Embeddings + Neural Networks (Predictive Maintenance)

**Cel:** pokazać nowoczesne podejście do ML na danych produkcyjnych (AI4I 2020, predictive maintenance):
- kategorie → **embeddingi** (zamiast one-hot),
- liczby → standaryzacja,
- model główny → **sieć neuronowa** dla danych tabelarycznych,
- baseline → **XGBoost** do porównania.

Używamy zbioru **AI4I 2020 – Predictive Maintenance** (UCI).  
Plik `ai4i2020.csv` umieść w `data/raw/`.

Kolumny przykładowe:
- Kategoryczne: `Product ID`, `Type`
- Numeryczne: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- Target (klasyfikacja): `Machine failure` (0/1) + składowe przyczyn awarii

> Możesz łatwo podmienić target i zrobić **regresję** (np. przewidywanie `Tool wear [min]`), bez zmiany struktury projektu.
