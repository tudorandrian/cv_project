import os
import shutil
import requests
import random
from tqdm import tqdm

# URL-ul bazei de date LFW
lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
lfw_archive = "lfw.tgz"
lfw_extract_dir = "lfw"
output_dir = "dataset_limited"

# Descărcare baza de date
if not os.path.exists(lfw_archive):
    print("Descărcare baza de date LFW...")
    response = requests.get(lfw_url, stream=True)
    with open(lfw_archive, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Descărcare"):
            file.write(chunk)

# Dezarhivare baza de date
if not os.path.exists(lfw_extract_dir):
    print("Dezarhivare arhivă...")
    import tarfile
    with tarfile.open(lfw_archive, "r:gz") as tar:
        tar.extractall()

# Ștergere vechilor imagini (dacă există)
if os.path.exists(output_dir):
    print("Ștergere imagini vechi...")
    shutil.rmtree(output_dir)

# Creare director pentru noile imagini
os.makedirs(output_dir, exist_ok=True)

# Limitare la maxim 50 de persoane și 10 imagini per persoană
person_count = 0
max_persons = 50  # Maxim persoane
max_images = 10   # Maxim imagini per persoană

# Obține lista de persoane și aplică o ordine aleatorie
persons = sorted(os.listdir(lfw_extract_dir))
random.shuffle(persons)  # Shuffle aleatoriu

print("Organizare imagini...")
for person_dir in persons:
    if person_count >= max_persons:
        break

    # Creează director pentru persoană
    person_path = os.path.join(lfw_extract_dir, person_dir)
    if os.path.isdir(person_path):
        person_output_dir = os.path.join(output_dir, person_dir)
        os.makedirs(person_output_dir, exist_ok=True)

        # Copiază imagini limitate
        image_count = 0
        for image_name in sorted(os.listdir(person_path)):
            if image_count >= max_images:
                break
            image_path = os.path.join(person_path, image_name)
            output_path = os.path.join(person_output_dir, image_name)
            with open(image_path, "rb") as src, open(output_path, "wb") as dest:
                dest.write(src.read())
            image_count += 1

        person_count += 1

print(f"Imagini organizate: {person_count} persoane, {max_images} imagini per persoană.")
