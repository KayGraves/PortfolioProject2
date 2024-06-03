import os
import re
import collections
import math
import argparse

# Preprocess text
def preprocess(line):
    line = line.rstrip().lower()
    line = re.sub("[^a-z ]", '', line)
    tokens = line.split()
    tokens = ['$' + token + '$' for token in tokens]
    return tokens

# Create language model
def create_model(path):
    unigrams = collections.defaultdict(int)
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(path, 'r') as f:
        for l in f.readlines()[500:600]:
            tokens = preprocess(l)
            if len(tokens) == 0:
                continue
            for token in tokens[0:50]:
                for character in range(len(token) - 1):
                    current_character = token[character:character+1]
                    unigrams[current_character] += 1
                    bigram = token[character:character+2]
                    bigrams[bigram] += 1
    return bigrams, unigrams

# Predict language
def predict(file, model_en, model_es):
    bigrams = collections.defaultdict(int)
    unigrams = collections.defaultdict(int)
    with open(file, 'r') as f:
        for l in f.readlines()[20:500]:
            tokens = preprocess(l)
            if len(tokens) == 0:
                continue
            for token in tokens:
                for character in range(len(token) - 1):
                    current_character = token[character:character+1]
                    unigrams[current_character] += 1
                    bigram = token[character:character+2]
                    bigrams[bigram] += 1
    english_prob = calculate_probability(model_en, unigrams)
    spanish_prob = calculate_probability(model_es, unigrams)
    prediction = 'English' if english_prob < spanish_prob else 'Spanish'
    return prediction

# Calculate smoothed log probabilities
def calculate_probability(model, unigrams):
    total_unigrams = sum(unigrams.values()) + 26
    probability = 0
    for bigram, count in model.items():
        if isinstance(count, str):
            continue
        count += 1
        x = bigram[0]
        x_count = unigrams[x] + 1
        probability += math.log(count / x_count)
    return probability

def main(en_tr, es_tr, folder_te):
    model_en = create_model(en_tr)
    model_es = create_model(es_tr)

    folder_en = os.path.join(folder_te, "en")
    print("Prediction for English documents in test:")
    for f in os.listdir(folder_en):
        f_path = os.path.join(folder_en, f)
        print(f"{f}\t{predict(f_path, model_en, model_es)}")

    folder_es = os.path.join(folder_te, "es")
    print("\nPrediction for Spanish documents in test:")
    for f in os.listdir(folder_es):
        f_path = os.path.join(folder_es, f)
        print(f"{f}\t{predict(f_path, model_en, model_es)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR_EN", help="Path to file with English training files")
    parser.add_argument("PATH_TR_ES", help="Path to file with Spanish training files")
    parser.add_argument("PATH_TEST", help="Path to folder with test files")
    args = parser.parse_args()

    main(args.PATH_TR_EN, args.PATH_TR_ES, args.PATH_TEST)
