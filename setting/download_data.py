import kagglehub

# Download latest version
path = kagglehub.dataset_download("hasibullahaman/traffic-prediction-dataset")

print("Path to dataset files:", path)