# Importiramo potrebnu klasu iz ultralytics biblioteke
from ultralytics import YOLO
import torch
import os
import sys
from pathlib import Path

def validiraj_model_objekt(model, data_path='yolo_dataset/data.yaml', verbose=True):
    """
    Validacija postojećeg YOLO model objekta.
    Ova funkcija se koristi iz train.py nakon treniranja.
    
    Args:
        model: YOLO model objekt (već učitan)
        data_path (str): Putanja do data.yaml datoteke
        verbose (bool): Treba li ispisivati detaljne informacije
    
    Returns:
        metrics: Rezultati validacije ili None ako je došlo do greške
    """
    try:
        if verbose:
            print("Pokretanje validacije na 'val' setu s najboljim modelom...")
        
        # Pokretanje validacije
        metrics = model.val(data=data_path)
        
        if verbose:
            print("--------------------------------------------------")
            print("Metrike validacije:")
            # Ispis metrika
            if hasattr(metrics, 'box') and metrics.box is not None:
                print(f"  Preciznost (Precision): {metrics.box.p[0]:.4f}")
                print(f"  Odaziv (Recall):        {metrics.box.r[0]:.4f}")
                print(f"  mAP50:                  {metrics.box.map50:.4f}")
                print(f"  mAP50-95:               {metrics.box.map:.4f}")
            else:
                print("Nije moguće automatski izvući metrike na ovaj način. Provjerite ispis iznad ili rezultate u direktoriju.")
            print("--------------------------------------------------")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"GREŠKA tijekom validacije: {e}")
        return None

def validiraj_model(model_path=None, data_path='yolo_dataset/data.yaml'):
    """
    Funkcija za validaciju YOLO modela na validation setu.
    Ova funkcija se koristi za nezavisnu validaciju iz command line-a.
    
    Args:
        model_path (str): Putanja do modela (.pt fajl). Ako nije specificirana, 
                         pokušava pronaći najbolji model iz najnovijeg treniranja.
        data_path (str): Putanja do data.yaml datoteke.
    """
    print("--------------------------------------------------")
    print("Pokretanje validacije YOLO modela...")
    print("--------------------------------------------------")
    
    # --- KORAK 1: Određivanje putanje do modela ---
    if model_path is None:
        # Pokušaj pronaći najbolji model iz najnovijeg treniranja
        runs_dir = Path('runs/detect')
        if runs_dir.exists():
            # Pronađi najnoviji direktorij treniranja
            training_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and 'trening' in d.name]
            if training_dirs:
                # Sortiraj po vremenu kreiranja (najnoviji prvi)
                latest_training = max(training_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_training / 'weights' / 'best.pt'
                if best_model_path.exists():
                    model_path = str(best_model_path)
                    print(f"Pronađen najbolji model iz najnovijeg treniranja: {model_path}")
                else:
                    print("Nije pronađen best.pt u najnovijem treniranju.")
            else:
                print("Nisu pronađeni direktoriji treniranja.")
        
        # Ako još uvijek nema model_path, pokušaj s osnovnim modelom
        if model_path is None:
            if Path('yolo11n.pt').exists():
                model_path = 'yolo11n.pt'
                print("Koristim osnovni yolo11n.pt model.")
            else:
                print("GREŠKA: Nije specificiran model i nije pronađen nijedan model za validaciju.")
                print("Molimo specificirajte putanju do modela ili pokrenite treniranje prvo.")
                return None
    
    # Provjeri postoji li specificiran model
    if not Path(model_path).exists():
        print(f"GREŠKA: Model na putanji '{model_path}' ne postoji.")
        return None
    
    # Provjeri postoji li data.yaml
    if not Path(data_path).exists():
        print(f"GREŠKA: Data.yaml datoteka na putanji '{data_path}' ne postoji.")
        return None
    
    print(f"Model za validaciju: {model_path}")
    print(f"Dataset konfiguracija: {data_path}")
    print("--------------------------------------------------")
    
    # --- KORAK 2: Provjera GPU-a ---
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU je dostupan: {device_name}")
        device = 0
    else:
        print("GPU nije dostupan, validacija će se izvršiti na CPU.")
        device = 'cpu'
    print("--------------------------------------------------")
    
    # --- KORAK 3: Učitavanje modela ---
    try:
        print("Učitavanje modela...")
        model = YOLO(model_path)
        print("Model uspješno učitan.")
    except Exception as e:
        print(f"GREŠKA pri učitavanju modela: {e}")
        return None
    
    # --- KORAK 4: Pokretanje validacije koristeći zajedničku funkciju ---
    return validiraj_model_objekt(model, data_path, verbose=True)

def main():
    """
    Glavna funkcija koja parsira argumente i pokreće validaciju.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Validacija YOLO modela')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Putanja do modela (.pt fajl). Ako nije specificirana, pokušava pronaći najbolji model.')
    parser.add_argument('--data', '-d', type=str, default='yolo_dataset/data.yaml',
                       help='Putanja do data.yaml datoteke (default: yolo_dataset/data.yaml)')
    
    args = parser.parse_args()
    
    # Pokreni validaciju
    metrics = validiraj_model(model_path=args.model, data_path=args.data)
    
    if metrics is None:
        sys.exit(1)  # Izađi s greškom ako validacija nije uspjela

if __name__ == '__main__':
    main()