import os
import pickle
import face_recognition

# Directorul cu imaginile descărcate și organizate
dataset_path = "dataset_limited"

# Listă pentru embeddings și numele persoanelor
known_encodings = []
known_names = []

print("Extragere embeddings faciali...")
# Parcurge persoanele din dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    # Parcurge imaginile fiecărei persoane
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Încarcă imaginea și extrage embeddings faciali
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # Dacă există embeddings validați
            known_encodings.append(encodings[0])
            known_names.append(person_name)

# Salvează embeddings și numele în fișier .pickle
output_file = "embeddings_faces.pickle"
with open(output_file, "wb") as file:
    pickle.dump({"encodings": known_encodings, "names": known_names}, file)

print(f"Embeddings salvați în {output_file}.")
