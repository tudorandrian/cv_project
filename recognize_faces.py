import os
import random
import cv2
import face_recognition
import pickle

# Fișierul embeddings generat anterior
embeddings_file = "embeddings_faces.pickle"

# Încarcă embeddings salvați
print("Încărcare embeddings...")
with open(embeddings_file, "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"Încărcate {len(known_encodings)} embeddings.")

# Directorul pentru imaginile procesate
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Funcție pentru recunoașterea fețelor și salvarea imaginilor procesate
def recognize_and_save_faces(image_path, person_name):
    print(f"Procesare imagine: {image_path}")
    # Încarcă imaginea
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectează locațiile fețelor și generează embeddings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Compară embeddings detectate cu cele cunoscute
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Necunoscut"

        # Dacă găsește un match
        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

        # Desenează dreptunghiul și numele persoanei pe imagine
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Creează subdirector pentru persoană în `processed_images`
    person_output_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_output_dir, exist_ok=True)

    # Salvează imaginea procesată în subdirectorul persoanei
    output_path = os.path.join(person_output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Imagine procesată salvată în {output_path}")
    return output_path  # Returnează calea către imaginea procesată

# Procesare pentru toate imaginile din dataset_limited
dataset_path = "dataset_limited"
processed_images = []  # Listă pentru imaginile procesate

print("Procesare toate imaginile din dataset_limited...")
for person_name in os.listdir(dataset_path):  # Parcurge subdirectoarele (persoane)
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue  # Ignoră fișierele care nu sunt directoare

    for image_name in os.listdir(person_path):  # Parcurge imaginile fiecărei persoane
        image_path = os.path.join(person_path, image_name)
        processed_path = recognize_and_save_faces(image_path, person_name)
        processed_images.append(processed_path)  # Adaugă imaginea procesată în listă

# Afișare maxim 5 imagini aleatoriu din cele procesate
print("Afișare imagini selectate aleatoriu...")

random.shuffle(processed_images)
for image_path in processed_images[:5]:
    image = cv2.imread(image_path)
    cv2.imshow("Imagine Procesată", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("Procesare completă.")
