#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITERATIVNO TRENIRANJE YOLO MODELA - ACV PRISTUP
===============================================

Ova skripta implementira iterativni pristup treniranju slično Azure Custom Vision:
- Svaka iteracija koristi najbolji model iz prethodne iteracije
- Postupno poboljšavanje kroz više kratkih treniranja
- Automatsko praćenje napretka između iteracija

Autor: AI Assistant
Datum: 2025
"""

import os
import glob
import json
from datetime import datetime
from ultralytics import YOLO
import torch

def pronadji_najbolji_model(runs_dir="runs/detect"):
    """
    Pronalazi najbolji model iz svih prethodnih treniranja.
    Vraća putanju do best.pt datoteke s najboljim mAP50.
    """
    najbolji_model = None
    najbolji_map50 = 0.0
    
    # Traži sve direktorije s treniranjima
    pattern = os.path.join(runs_dir, "*/weights/best.pt")
    model_paths = glob.glob(pattern)
    
    for model_path in model_paths:
        try:
            # Učitaj model i provjeri njegove rezultate
            model = YOLO(model_path)
            
            # Pokušaj pronaći results.json u istom direktoriju
            results_dir = os.path.dirname(os.path.dirname(model_path))
            results_file = os.path.join(results_dir, "results.csv")
            
            if os.path.exists(results_file):
                # Čitaj zadnji red iz results.csv (najbolji rezultat)
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Preskačemo header
                        last_line = lines[-1].strip().split(',')
                        # mAP50 je obično u 7. koloni (indeks 6)
                        if len(last_line) > 6:
                            try:
                                map50 = float(last_line[6])
                                if map50 > najbolji_map50:
                                    najbolji_map50 = map50
                                    najbolji_model = model_path
                                    print(f"Pronađen bolji model: {model_path} (mAP50: {map50:.4f})")
                            except (ValueError, IndexError):
                                continue
        except Exception as e:
            print(f"Greška pri čitanju modela {model_path}: {e}")
            continue
    
    return najbolji_model, najbolji_map50

def pokreni_iteraciju(iteracija_broj, base_model_path=None):
    """
    Pokreće jednu iteraciju treniranja.
    """
    print(f"\n{'='*60}")
    print(f"🚀 ITERACIJA {iteracija_broj}")
    print(f"{'='*60}")
    
    # Provjeri dostupnost GPU-a
    if torch.cuda.is_available():
        device = 0  # Koristi prvi GPU
        print(f"✅ GPU dostupan: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️  GPU nije dostupan, koristit će se CPU (sporije)")
    
    # Odaberi početni model
    if base_model_path and os.path.exists(base_model_path):
        print(f"📂 Koristim prethodni najbolji model: {base_model_path}")
        model = YOLO(base_model_path)
    else:
        print("🆕 Počinjem s YOLOv11m pretreniranim modelom (Medium)")
        model = YOLO('yolo11m.pt')
    
    # Naziv za ovu iteraciju
    run_name = f'iteracija_{iteracija_broj:02d}'
    
    print(f"🎯 Pokretanje treniranja: {run_name}")
    print("⏱️  Očekivano vrijeme: 1.5-2 sata (40 epoha, umjereno)")
    
    try:
        # Pokreni treniranje optimizirano za CPU s punom rezolucijom
        results = model.train(
            data='yolo_dataset/data.yaml',
            epochs=40,              # Umjereno smanjujem epohe
            imgsz=832,              # Kompromis između kvalitete i stabilnosti
            batch=4,                # Umjeren batch size
            name=run_name,
            device='cpu',           # CPU za korištenje RAM prednosti
            patience=12,            # Umjerena patience
            workers=4,              # Pola od maksimalnih workera
            cache=True,             # Zadržavam cache - imate 64GB!
            amp=False,              # Bez mixed precision na CPU
            save_period=10,         # Spremi checkpoint svakih 10 epoha
            plots=True,             # Generiraj grafove
            verbose=True
        )
        
        print(f"✅ Iteracija {iteracija_broj} završena uspješno!")
        return True, results
        
    except Exception as e:
        print(f"❌ Greška u iteraciji {iteracija_broj}: {e}")
        return False, None

def main():
    """
    Glavna funkcija za iterativno treniranje.
    """
    print("🔄 ITERATIVNO TRENIRANJE YOLO MODELA")
    print("Pristup sličan Azure Custom Vision")
    print("=" * 50)
    
    # Postavke
    max_iteracija = 5  # Maksimalno 5 iteracija
    min_poboljsanje = 0.01  # Minimalno poboljšanje mAP50 za nastavak
    
    najbolji_map50_ukupno = 0.0
    
    for iteracija in range(1, max_iteracija + 1):
        # Pronađi najbolji model iz prethodnih iteracija
        if iteracija > 1:
            najbolji_model, najbolji_map50 = pronadji_najbolji_model()
            
            if najbolji_model:
                print(f"📈 Trenutno najbolji mAP50: {najbolji_map50:.4f}")
                
                # Provjeri je li dovoljno poboljšanje
                if iteracija > 2 and (najbolji_map50 - najbolji_map50_ukupno) < min_poboljsanje:
                    print(f"⏹️  Prekidam - poboljšanje manje od {min_poboljsanje}")
                    break
                
                najbolji_map50_ukupno = najbolji_map50
            else:
                print("⚠️  Nema prethodnih modela, koristim pretreniran")
                najbolji_model = None
        else:
            najbolji_model = None
        
        # Pokreni iteraciju
        uspjeh, results = pokreni_iteraciju(iteracija, najbolji_model)
        
        if not uspjeh:
            print(f"❌ Prekidam zbog greške u iteraciji {iteracija}")
            break
        
        print(f"✅ Iteracija {iteracija} završena")
        
        # Kratka pauza između iteracija
        import time
        time.sleep(2)
    
    # Finalni rezultati
    print(f"\n{'='*60}")
    print("🏁 ITERATIVNO TRENIRANJE ZAVRŠENO")
    print(f"{'='*60}")
    
    finalni_model, finalni_map50 = pronadji_najbolji_model()
    if finalni_model:
        print(f"🏆 Najbolji model: {finalni_model}")
        print(f"📊 Finalni mAP50: {finalni_map50:.4f}")
        
        # Pokreni finalnu validaciju
        print("\n🔍 Pokretanje finalne validacije...")
        try:
            from validate import validiraj_model_objekt
            model = YOLO(finalni_model)
            validiraj_model_objekt(model, 'yolo_dataset/data.yaml')
        except Exception as e:
            print(f"⚠️  Greška pri finalnoj validaciji: {e}")
    else:
        print("❌ Nema uspješnih modela")

if __name__ == "__main__":
    main()