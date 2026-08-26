import pickle
import os

folder = "/home/kenzosaki/repos/glm-based-event-analysis/data/raw/gnews_results"
total = 0
for file in os.listdir(folder):
    if file.endswith(".pkl"):
        file_path = os.path.join(folder, file)  
        with open(file_path, "rb") as f:
            news_data = pickle.load(f)
            print(f"File: {file}, Number of news articles: {sum((len(batch) for batch in news_data))}")
            total += sum((len(batch) for batch in news_data))
print(f"Total number of news articles: {total}")