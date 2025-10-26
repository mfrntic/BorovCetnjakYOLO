# Importiramo potrebnu klasu iz ultralytics biblioteke
from ultralytics import YOLO
import torch # Importiramo PyTorch za provjeru GPU-a (opcionalno)
from validate import validiraj_model_objekt  # Importiramo funkciju za validaciju

def treniraj_model():
    """
    Funkcija za treniranje najnovijeg YOLO modela (YOLOv11) na custom datasetu.
    """
    print("--------------------------------------------------")
    print("Provjera dostupnosti GPU-a...")
    # Provjeravamo je li GPU dostupan (opcionalno, ali korisno za praćenje)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU je dostupan: {device_name}")
        device = 0 # Koristi prvi GPU (indeks 0)
    else:
        print("GPU nije dostupan, treniranje će se izvršiti na CPU (bit će sporije).")
        device = 'cpu' # Koristi CPU
    print("--------------------------------------------------")

    # --- KORAK 1: Učitavanje pre-treniranog modela ---
    # Koristimo najveći dostupni YOLO11 model za najbolje rezultate.
    # yolo11x.pt je najveći model s najboljom točnošću, ali zahtijeva više resursa.
    # Model će se automatski skinuti ako ne postoji lokalno.
    print("Učitavanje pre-treniranog modela (yolo11x.pt - najveći model)...")
    model = YOLO('yolo11x.pt')  # Najveći i najnapredniji model
    print("Model učitan.")
    print("--------------------------------------------------")

    # --- KORAK 2: Pokretanje treniranja ---
    # Ovdje specificiramo putanju do 'data.yaml' datoteke i parametre treniranja.
    print("Pokretanje treniranja...")
    try:
        results = model.train(
            data='yolo_dataset/data.yaml',  # *** VAŽNO: Zamijenite ovo s točnom putanjom do vaše data.yaml datoteke ***
            epochs=50,          # Smanjeno na 50 - dovoljno za transfer learning
            imgsz=640,          # Vraćeno na 640 - brže treniranje
            batch=16,           # Povećano na 16 - brže ako GPU može podnijeti
            name='yolo11x_gnijezda_brzo', # Naziv za brže treniranje
            device=device,      # Eksplicitno kažemo da koristi GPU (0) ili CPU ('cpu')
            patience=15,        # Smanjeno na 15 - brže zaustavljanje
            workers=8,          # Povećano za brže učitavanje podataka
            cache=True,         # Cache slike u RAM-u za brže učitavanje
            amp=True,           # Mixed precision - brže treniranje
            # project='neki_drugi_direktorij' # (Opcionalno) Ako ne želite da se rezultati spremaju u 'runs/detect/'
        )
        print("--------------------------------------------------")
        print("Treniranje završeno uspješno.")
        print(f"Rezultati spremljeni u direktorij: {results.save_dir}") # Ispisuje točnu putanju do rezultata
        print("--------------------------------------------------")

        # --- KORAK 3: Validacija najboljeg modela ---
        # Nakon treniranja, automatski se koristi najbolji model ('best.pt') za validaciju.
        # Koristimo zajedničku funkciju iz validate.py da izbjegnemo dupliciranje koda
        metrics = validiraj_model_objekt(model, data_path='yolo_dataset/data.yaml')


    except Exception as e:
        print(f"Došlo je do greške tijekom treniranja ili validacije: {e}")

# --- Glavni dio skripte ---
if __name__ == '__main__':
    # Ova provjera osigurava da se funkcija treniraj_model() pokreće
    # samo kada se skripta izvršava direktno (ne kada se importira kao modul).
    treniraj_model()