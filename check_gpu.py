import torch
import sys

print("=== GPU STATUS PROVJERA ===")
print(f"PyTorch verzija: {torch.__version__}")
print(f"CUDA dostupan: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA verzija: {torch.version.cuda}")
    print(f"Broj GPU-ova: {torch.cuda.device_count()}")
    print(f"GPU naziv: {torch.cuda.get_device_name(0)}")
    print(f"GPU memorija: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Trenutno korištena memorija: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
else:
    print("❌ CUDA NIJE DOSTUPAN!")
    print("Mogući razlozi:")
    print("1. PyTorch instaliran bez CUDA podrške")
    print("2. NVIDIA driveri nisu instalirani")
    print("3. CUDA toolkit nije instaliran")
    print("4. Nekompatibilna verzija CUDA/PyTorch")

# Provjeri ultralytics
try:
    from ultralytics import YOLO
    print(f"\n✅ Ultralytics uspješno importiran")
    
    # Test YOLO model
    model = YOLO('yolo11n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"YOLO će koristiti: {device}")
    
except Exception as e:
    print(f"❌ Greška s Ultralytics: {e}")