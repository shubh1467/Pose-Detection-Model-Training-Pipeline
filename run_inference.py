import cv2
import numpy as np
from ultralytics import YOLO
import joblib

# =========================
# LOAD MODELS
# =========================
pose_model = YOLO("yolo11x-pose.pt")
model = joblib.load("pose_model.pkl")

# 🔥 LOAD LABEL MAPS
remap = joblib.load("label_remap.pkl")              # {old: new}
inverse_map = {v: k for k, v in remap.items()}      # {new: old}

label_names = {
    0: "FRONT",
    1: "SIDE",
    2: "WRONG"
}

# =========================
# FEATURE FUNCTION (SAME AS TRAINING)
# =========================
def extract_features(kps):
    def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))
    def visible(kps,i): return kps[i][2]>0.5
    def angle(p1,p2,p3):
        a = np.array(p1)-np.array(p2)
        b = np.array(p3)-np.array(p2)
        cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6)
        return np.degrees(np.arccos(np.clip(cos,-1,1)))

    L_SHOULDER,R_SHOULDER=5,6
    L_HIP,R_HIP=11,12
    NOSE,LEFT_EYE,RIGHT_EYE=0,1,2
    LEFT_EAR,RIGHT_EAR=3,4
    L_ELBOW,R_ELBOW=7,8
    L_WRIST,R_WRIST=9,10
    L_KNEE,R_KNEE=13,14
    L_ANKLE,R_ANKLE=15,16

    if not (visible(kps,L_SHOULDER) and visible(kps,R_SHOULDER) and
            visible(kps,L_HIP) and visible(kps,R_HIP)):
        return None

    shoulder = dist(kps[L_SHOULDER][:2], kps[R_SHOULDER][:2])
    hip = dist(kps[L_HIP][:2], kps[R_HIP][:2])

    sm = ((kps[L_SHOULDER][0]+kps[R_SHOULDER][0])/2,
          (kps[L_SHOULDER][1]+kps[R_SHOULDER][1])/2)

    hm = ((kps[L_HIP][0]+kps[R_HIP][0])/2,
          (kps[L_HIP][1]+kps[R_HIP][1])/2)

    torso = dist(sm, hm)

    eyes = int(visible(kps,LEFT_EYE)) + int(visible(kps,RIGHT_EYE))
    ears = int(visible(kps,LEFT_EAR)) + int(visible(kps,RIGHT_EAR))
    nose = int(visible(kps,NOSE))

    nose_sym = 0
    if eyes == 2:
        dl = dist(kps[NOSE][:2], kps[LEFT_EYE][:2])
        dr = dist(kps[NOSE][:2], kps[RIGHT_EYE][:2])
        nose_sym = min(dl,dr)/(max(dl,dr)+1e-6)

    shoulder_angle = abs(np.degrees(np.arctan2(
        kps[R_SHOULDER][1] - kps[L_SHOULDER][1],
        kps[R_SHOULDER][0] - kps[L_SHOULDER][0]
    )))

    spine_angle = abs(np.degrees(np.arctan2(
        hm[0] - sm[0],
        hm[1] - sm[1]
    )))

    asymmetry = abs(kps[L_SHOULDER][1] - kps[R_SHOULDER][1]) / (shoulder+1e-6)

    left_elbow = angle(kps[L_SHOULDER][:2], kps[L_ELBOW][:2], kps[L_WRIST][:2]) \
        if visible(kps,L_ELBOW) else 0

    right_elbow = angle(kps[R_SHOULDER][:2], kps[R_ELBOW][:2], kps[R_WRIST][:2]) \
        if visible(kps,R_ELBOW) else 0

    left_knee = angle(kps[L_HIP][:2], kps[L_KNEE][:2], kps[L_ANKLE][:2]) \
        if visible(kps,L_KNEE) else 0

    right_knee = angle(kps[R_HIP][:2], kps[R_KNEE][:2], kps[R_ANKLE][:2]) \
        if visible(kps,R_KNEE) else 0

    return [
        shoulder, hip, torso,
        shoulder/(hip+1e-6),
        shoulder/(torso+1e-6),
        eyes, ears, nose,
        nose_sym,
        shoulder_angle,
        spine_angle,
        asymmetry,
        left_elbow, right_elbow,
        left_knee, right_knee
    ]

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture("hit1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = pose_model(frame)[0]

    label_text = "UNKNOWN"

    if res.keypoints is not None:
        kps = res.keypoints.data[0].cpu().numpy()
        feats = extract_features(kps)

        if feats is not None:
            pred = model.predict([feats])[0]        # e.g. 0 or 1

            # 🔥 Convert back to original label
            original_label = inverse_map[pred]      # e.g. 0 or 2

            # 🔥 Convert to string name
            label_text = label_names[original_label]

    # 🔥 ALWAYS STRING
    cv2.putText(frame, str(label_text), (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()