import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Załadowanie modelu
model = tf.keras.models.load_model('dog_breed_classifier.keras')

# Lista ras psów w języku angielskim
dog_breeds = [
    "Affenpinscher", "Afghan Hound", "Airedale Terrier", "Akita", "Alaskan Malamute", "American Eskimo Dog",
    "American Foxhound", "American Staffordshire Terrier", "American Water Spaniel", "Anatolian Shepherd Dog",
    "Australian Cattle Dog", "Australian Shepherd", "Australian Terrier", "Basenji", "Basset Hound", "Beagle",
    "Bearded Collie", "Beauceron", "Bedlington Terrier", "Belgian Malinois", "Belgian Sheepdog", "Belgian Tervuren",
    "Bernese Mountain Dog", "Bichon Frise", "Black and Tan Coonhound", "Black Russian Terrier", "Bloodhound",
    "Bluetick Coonhound", "Border Collie", "Border Terrier", "Borzoi", "Boston Terrier", "Bouvier des Flandres",
    "Boxer", "Boykin Spaniel", "Briard", "Brittany", "Brussels Griffon", "Bull Terrier", "Bulldog", "Bullmastiff",
    "Cairn Terrier", "Canaan Dog", "Cane Corso", "Cardigan Welsh Corgi", "Cavalier King Charles Spaniel",
    "Chesapeake Bay Retriever",
    "Chihuahua", "Chinese Crested", "Chinese Shar-Pei", "Chow Chow", "Clumber Spaniel", "Cocker Spaniel", "Collie",
    "Curly-Coated Retriever", "Dachshund", "Dalmatian", "Dandie Dinmont Terrier", "Doberman Pinscher",
    "Dogue de Bordeaux",
    "English Cocker Spaniel", "English Setter", "English Springer Spaniel", "English Toy Spaniel",
    "Entlebucher Mountain Dog",
    "Field Spaniel", "Finnish Spitz", "Flat-Coated Retriever", "French Bulldog", "German Pinscher",
    "German Shepherd Dog",
    "German Shorthaired Pointer", "German Wirehaired Pointer", "Giant Schnauzer", "Glen of Imaal Terrier",
    "Golden Retriever",
    "Gordon Setter", "Great Dane", "Great Pyrenees", "Greater Swiss Mountain Dog", "Greyhound", "Havanese",
    "Ibizan Hound", "Icelandic Sheepdog", "Irish Red and White Setter", "Irish Setter", "Irish Terrier",
    "Irish Water Spaniel", "Irish Wolfhound", "Italian Greyhound", "Japanese Chin", "Keeshond", "Kerry Blue Terrier",
    "Komondor", "Kuvasz", "Labrador Retriever", "Lakeland Terrier", "Leonberger", "Lhasa Apso", "Lowchen", "Maltese",
    "Manchester Terrier", "Mastiff", "Miniature Schnauzer", "Neapolitan Mastiff", "Newfoundland", "Norfolk Terrier",
    "Norwegian Buhund", "Norwegian Elkhound", "Norwegian Lundehund", "Norwich Terrier",
    "Nova Scotia Duck Tolling Retriever",
    "Old English Sheepdog", "Otterhound", "Papillon", "Parson Russell Terrier", "Pekingese", "Pembroke Welsh Corgi",
    "Petit Basset Griffon Vendeen", "Pharaoh Hound", "Plott", "Pointer", "Pomeranian", "Poodle", "Portuguese Water Dog",
    "Saint Bernard", "Silky Terrier", "Smooth Fox Terrier", "Tibetan Mastiff", "Welsh Springer Spaniel",
    "Wirehaired Pointing Griffon", "Xoloitzcuintli", "Yorkshire Terrier"
]


# Funkcja do ładowania i przetwarzania obrazu
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Wczytanie ścieżki do obrazu z argumentów wiersza poleceń
if len(sys.argv) < 2:
    print("Usage: python test_model.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
img = load_and_preprocess_image(img_path)

# Predykcja
predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_breed = dog_breeds[predicted_class]
confidence = predictions[0][predicted_class] * 100

# Sortowanie wyników i wybranie trzech najwyższych
top_3_indices = np.argsort(predictions[0])[-3:][::-1]
top_3_breeds = [(dog_breeds[i], predictions[0][i] * 100) for i in top_3_indices]

# Wypisanie wyniku
print(f'Predykowana klasa: {predicted_breed} ({confidence:.2f}%)')

# Wypisanie trzech najwyższych prawdopodobieństw
print("\nTrzy najwyższe prawdopodobieństwa:")
for breed, prob in top_3_breeds:
    print(f"{breed}: {prob:.2f}%")
