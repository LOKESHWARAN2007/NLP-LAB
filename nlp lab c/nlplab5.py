import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Input text
text = input("Enter legal text: ")

# Tokenize and tag
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print("\nDetected Named Entities:")
count = 0
for word, tag in tags:
    if tag == "NNP":  # Proper noun, singular
        print(word, "-> ENTITY")
        count += 1

# Accuracy calculation
actual = int(input("\nEnter actual number of entities: "))
accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")
