import torch, numpy as np
from torchvision.models import efficientnet_b0
from torchvision import transforms
import torch.nn as nn

ckpt = torch.load('models/video_classifier/best_webcam_model.pth', map_location='cpu', weights_only=False)
print('=== FINAL MODEL INFO ===')
print(f"Architecture : {ckpt.get('architecture','?')}")
print(f"Val Accuracy : {ckpt.get('val_acc',0):.2f}%")
print(f"Best Epoch   : {ckpt.get('epoch','?')}")
print(f"Image Size   : {ckpt.get('img_size','?')}px")
print(f"Labels       : {ckpt['label_mapping']}")

m = efficientnet_b0(weights=None)
inf = m.classifier[1].in_features
m.classifier = nn.Sequential(
    nn.Dropout(0.35, True), nn.Linear(inf, 256), nn.SiLU(),
    nn.Dropout(0.25), nn.Linear(256, 7)
)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()

n = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
lbl = {v:k for k,v in ckpt['label_mapping'].items()}

print()
print('=== GREY FRAME TEST (blank 128,128,128 — no face) ===')
img = np.ones((96,96,3), dtype='uint8') * 128
t = torch.from_numpy(img).permute(2,0,1).float()/255.0
t = n(t).unsqueeze(0)
with torch.no_grad():
    l = m(t)[0]; p = torch.softmax(l, 0)
for i in range(7):
    bar = '#' * int(p[i].item()*50)
    print(f"  {lbl[i]:10s}: {l[i].item():+.3f}  {p[i].item()*100:5.1f}%  {bar}")
winner = lbl[int(p.argmax())]
print(f"\n  Predicted  : {winner.upper()} ({p.max().item()*100:.1f}%)")
print(f"\n  OLD model  : JOY 41.9%  (completely wrong for blank grey)")
print(f"  NEW model  : {winner.upper()} {p.max().item()*100:.1f}%  (max conf drop = good!)")
print()
print('=== WHAT THIS MEANS FOR WEBCAM ===')
print(f"  Max raw logit spread: {l.max().item() - l.min().item():.3f}  (smaller = less biased)")
print(f"  Old model spread was: ~3.5  (JOY dominated everything)")
