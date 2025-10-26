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

def find_last_checkpoint(run_name):
    """
    Pronađi zadnji checkpoint iz prethodnog treniranja
    """
    run_path = f'runs/detect/{run_name}'
    weights_path = f'{run_path}/weights'
    
    if not os.path.exists(weights_path):
        return None, 0
    
    # Provjeri postoji li last.pt
    last_pt = f'{weights_path}/last.pt'
    if os.path.exists(last_pt):
        # Čitaj results.csv da vidiš do koje epohe je stigao
        results_csv = f'{run_path}/results.csv'
        if os.path.exists(results_csv):
            try:
                # Čitaj CSV bez pandas-a
                with open(results_csv, 'r') as f:
                    lines = f.readlines()
                    # Prvi red je header, ostali su epohe
                    last_epoch = len(lines) - 1  # -1 jer je prvi red header
                    if last_epoch > 0:
                        print(f"📊 Pronašao last.pt - zadnja završena epoha: {last_epoch}")
                        return last_pt, last_epoch
                    else:
                        print("⚠️  results.csv je prazan")
                        return last_pt, 0
            except Exception as e:
                print(f"⚠️  Greška pri čitanju results.csv: {e}")
                return last_pt, 0
    
    return None, 0

def pokreni_iteraciju(iteracija_broj, base_model_path=None, resume=False):
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
    
    # Naziv za ovu iteraciju
    run_name = f'iteracija_{iteracija_broj:02d}'
    
    # Provjeri možemo li nastaviti prethodno treniranje
    checkpoint_path, last_epoch = find_last_checkpoint(run_name)
    
    if resume and checkpoint_path and last_epoch > 0:
        print(f"🔄 NASTAVLJAM treniranje od epohe {last_epoch + 1}")
        print(f"📂 Koristim checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        start_epoch = last_epoch
    else:
        # Odaberi početni model
        if base_model_path and os.path.exists(base_model_path):
            print(f"📂 Koristim prethodni najbolji model: {base_model_path}")
            model = YOLO(base_model_path)
        else:
            print("🆕 Počinjem s YOLOv11s pretreniranim modelom (Small)")
            model = YOLO('yolo11s.pt')
        start_epoch = 0
    
    print(f"🎯 Pokretanje treniranja: {run_name}")
    print("⏱️  Očekivano vrijeme treniranja: 60-80 minuta za 40 epoha (imgsz=1280)")
    
    try:
        # Izračunaj preostale epohe
        total_epochs = 40
        remaining_epochs = total_epochs - start_epoch
        
        if remaining_epochs <= 0:
            print(f"✅ Treniranje već završeno! ({start_epoch}/{total_epochs} epoha)")
            return True, None
        
        print(f"📈 Treniram {remaining_epochs} preostalih epoha (od {start_epoch + 1} do {total_epochs})")
        
        # Pokreni treniranje optimizirano za CPU s punom rezolucijom
        results = model.train(
            data='yolo_dataset/data.yaml',
            epochs=remaining_epochs,    # Samo preostale epohe
            imgsz=1280,                 # Optimizirano za velike slike (2560x1709)
            batch=2,                    # Smanjen zbog većeg imgsz=1280
            project='runs/detect',      # Eksplicitno postavi project
            name=run_name,
            device='cpu',               # CPU za korištenje RAM prednosti
            patience=25,                # Povećana patience za small model
            workers=4,                  # Pola od maksimalnih workera
            cache=True,                 # Zadržavam cache - imate 64GB!
            amp=False,                  # Bez mixed precision na CPU
            save_period=10,             # Spremi checkpoint svakih 10 epoha
            plots=True,                 # Generiraj grafove
            verbose=True,
            exist_ok=True,              # Dozvoli postojanje direktorija
            resume=False                # Ne koristimo YOLO resume, već naš custom
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
    
    # Provjeri treba li nastaviti postojeće treniranje
    resume_option = input("🔄 Želite li nastaviti postojeće treniranje? (y/n): ").lower().strip()
    resume = resume_option in ['y', 'yes', 'da']
    
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
                
                # Provjeri je li poboljšanje dovoljno za nastavak
                poboljsanje = najbolji_map50 - najbolji_map50_ukupno
                if poboljsanje < min_poboljsanje:
                    print(f"⏹️  Poboljšanje ({poboljsanje:.4f}) je manje od minimuma ({min_poboljsanje})")
                    print("🏁 Završavam iterativno treniranje")
                    break
                
                najbolji_map50_ukupno = najbolji_map50
                base_model = najbolji_model
            else:
                base_model = None
        else:
            base_model = None
        
        # Pokreni iteraciju s resume opcijom
        uspjeh, rezultati = pokreni_iteraciju(iteracija, base_model, resume=resume)
        
        # Nakon prve iteracije, ne koristimo više resume
        resume = False
        
        if not uspjeh:
            print(f"❌ Iteracija {iteracija} neuspješna!")
            break
        
        print(f"✅ Iteracija {iteracija} završena!")
    
    print("\n🎉 ITERATIVNO TRENIRANJE ZAVRŠENO!")
    
    # Prikaži finalne rezultate
    najbolji_model, najbolji_map50 = pronadji_najbolji_model()
    if najbolji_model:
        print(f"🏆 Najbolji model: {najbolji_model}")
        print(f"📊 Najbolji mAP50: {najbolji_map50:.4f}")
    else:
        print("⚠️  Nema uspješno treniranih modela")

if __name__ == "__main__":
    main()