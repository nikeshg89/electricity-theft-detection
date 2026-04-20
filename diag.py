import sys, os, json
import numpy as np
import joblib

MODEL_DIR = 'model'

normal   = np.array([1.2,1.0,0.9,0.8,0.8,1.0,1.8,3.2,3.5,3.0,2.8,2.9,3.0,2.9,2.8,3.1,3.8,4.5,4.8,4.5,3.9,3.1,2.2,1.5])
theft    = np.array([0.05]*24)
nighthvy = np.array([12.0,11.0,10.5,10.0,9.5,9.0,0.2,0.2,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.3,0.2,0.3,8.0,10.0,11.0,12.0])

def profile_to_features(p):
    hours = np.arange(24)
    mean_c = p.mean()
    std_c = p.std()
    max_c = p.max()
    min_c = p.min()
    max_min = max_c - min_c
    night_mask = (hours < 6) | (hours >= 22)
    day_mask = ~night_mask
    night_sum = p[night_mask].sum()
    day_sum = p[day_mask].sum()
    nd_ratio = night_sum / (day_sum + 1e-6)
    peak_hour = float(np.argmax(p))
    flatline = float((p < 0.1).sum()) / 24.0
    spike = float((p > 2 * mean_c + 1e-6).sum()) / 24.0
    return np.array([mean_c, std_c, max_c, min_c, max_min, night_sum, day_sum, nd_ratio, peak_hour, flatline, spike])

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
clf = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
with open(os.path.join(MODEL_DIR, 'thresholds.json')) as f:
    thresh = json.load(f)

print("Thresholds:", thresh)
print()

for name, p in [("Normal", normal), ("Theft Flatline", theft), ("Night Heavy", nighthvy)]:
    feat = profile_to_features(p).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    score = float(clf.decision_function(feat_scaled)[0])
    nm = thresh['normal_min']
    sm = thresh['suspicious_max']
    if score >= nm:
        label = "NORMAL"
    elif score >= sm:
        label = "SLIGHTLY UNUSUAL"
    else:
        label = "SUSPICIOUS"
    print(f"{name}: score={score:.4f} normal_min={nm:.4f} -> {label}")
