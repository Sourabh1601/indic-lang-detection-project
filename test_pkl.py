import pickle

files = ['models/svm_model.pkl', 'models/scaler.pkl', 'models/rf_model.pkl', 'models/label_encoder.pkl']
for file in files:
    try:
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            print(f"{file} loaded successfully")
    except Exception as e:
            print(f"Error loading {file}: {e}")

