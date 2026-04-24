import cv2
import numpy as np
from ultralytics import YOLO
import csv
import math

# =========================
# LOAD MODELS
# =========================
pose_model = YOLO("yolo11x-pose.pt")

bat_model = YOLO("best.pt")
BAT_CLASS_ID = 0

video_input = "vid.mp4"

# =========================
# NEW: IOU FUNCTION (ADDED)
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# =========================
# KEYPOINTS (UNCHANGED)
# =========================
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

# =========================
# SKELETON (UNCHANGED)
# =========================
SKELETON = [
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

# =========================
# HELPERS (UNCHANGED)
# =========================
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def visible(kps, i):
    return kps[i][2] > 0.5

# =========================
# DRAW FUNCTIONS (UNCHANGED)
# =========================
def draw_pose(frame, kps):
    for kp in kps:
        x, y, c = kp
        if c > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

    for a, b in SKELETON:
        if kps[a][2] > 0.5 and kps[b][2] > 0.5:
            p1 = (int(kps[a][0]), int(kps[a][1]))
            p2 = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(frame, p1, p2, (0,255,0), 2)

# =========================
# UPDATED BAT FUNCTION
# =========================
def get_bat_info(result):
    if result.boxes is None:
        return None, None

    for box in result.boxes:
        if int(box.cls[0]) == BAT_CLASS_ID:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            center = np.array([(x1+x2)/2, (y1+y2)/2])
            return center, [x1,y1,x2,y2]

    return None, None


# =========================
# NEW: TORSO CENTER FUNCTION
# =========================
def get_torso_center(kps):
    if visible(kps, L_SHOULDER) and visible(kps, R_SHOULDER) and \
       visible(kps, L_HIP) and visible(kps, R_HIP):

        sm = (
            (kps[L_SHOULDER][0] + kps[R_SHOULDER][0]) / 2,
            (kps[L_SHOULDER][1] + kps[R_SHOULDER][1]) / 2
        )
        hm = (
            (kps[L_HIP][0] + kps[R_HIP][0]) / 2,
            (kps[L_HIP][1] + kps[R_HIP][1]) / 2
        )

        return ((sm[0] + hm[0]) / 2, (sm[1] + hm[1]) / 2)

    return None


# =========================
# UPDATED PERSON SCORE (VERY STRONG)
# =========================
def get_person_score(kps, person_box, bat_center, bat_box):

    if bat_center is None:
        return 999999

    score = 0

    # ======================
    # 1. WRIST DISTANCE (PRIMARY SIGNAL)
    # ======================
    wrists = []
    if visible(kps, L_WRIST):
        wrists.append(kps[L_WRIST][:2])
    if visible(kps, R_WRIST):
        wrists.append(kps[R_WRIST][:2])

    if len(wrists) > 0:
        wrist_dist = min(np.linalg.norm(np.array(w) - bat_center) for w in wrists)
    else:
        wrist_dist = 999999

    score += wrist_dist * 1.0   # 🔥 increase importance


    # ======================
    # 2. WRIST-TO-WRIST DISTANCE (NEW 🔥🔥🔥)
    # ======================
    # batsman holds bat → wrists close
    if visible(kps, L_WRIST) and visible(kps, R_WRIST):
        wrist_gap = np.linalg.norm(
            np.array(kps[L_WRIST][:2]) - np.array(kps[R_WRIST][:2])
        )
        score += wrist_gap * 0.5   # keeper hands usually far apart


    # ======================
    # 3. TORSO DISTANCE (REDUCED)
    # ======================
    torso = get_torso_center(kps)
    if torso is not None:
        torso_dist = np.linalg.norm(np.array(torso) - bat_center)
        score += torso_dist * 0.2   # 🔥 reduced weight


    # ======================
    # 4. Y-AXIS FILTER (VERY IMPORTANT 🔥)
    # ======================
    # keeper is usually higher (farther away)
    if torso is not None:
        if torso[1] < bat_center[1]:   # above bat
            score += 120   # penalize keeper
        else:
            score -= 40    # reward batsman


    # ======================
    # 5. IoU (KEEP BUT LIGHT)
    # ======================
    if bat_box is not None:
        iou = compute_iou(person_box, bat_box)
        score -= iou * 100


    return score

    # ======================
    # 2. TORSO DISTANCE (NEW 🔥)
    # ======================
    torso = get_torso_center(kps)
    if torso is not None:
        torso_dist = np.linalg.norm(np.array(torso) - bat_center)
        score += torso_dist * 0.8   # stronger than wrist


    # ======================
    # 3. IoU BONUS (existing)
    # ======================
    if bat_box is not None:
        iou = compute_iou(person_box, bat_box)
        score -= iou * 300   # stronger boost


    # ======================
    # 4. FRONT-OF-BAT FILTER (NEW 🔥)
    # ======================
    # Batsman is usually slightly in front of bat (x-direction)
    if torso is not None:
        if torso[0] < bat_center[0]:  
            score -= 50   # reward
        else:
            score += 50   # penalize (keeper often behind)


    # ======================
    # 5. DISTANCE REJECTION (NEW 🔥)
    # ======================
    # Reject far-away players (keeper often farther)
    if torso is not None:
        if torso_dist > 250:   # tune if needed
            score += 300


    return score
# =========================
# FEATURE EXTRACTION (UNCHANGED)
# =========================
def extract_features(kps):
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

    shoulder_angle = abs(math.degrees(math.atan2(
        kps[R_SHOULDER][1] - kps[L_SHOULDER][1],
        kps[R_SHOULDER][0] - kps[L_SHOULDER][0]
    )))

    spine_angle = abs(math.degrees(math.atan2(
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
# CSV SETUP (UNCHANGED)
# =========================
csv_file = open("pose_dataset.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "shoulder","hip","torso","sh_hip","sh_torso", "eyes",
    "ears","nose","nose_symmetry", "shoulder_angle","spine_angle",
    "asymmetry", "left_elbow","right_elbow", "left_knee","right_knee", "label"
])

# =========================
# VIDEO LOOP (UPDATED SELECTION ONLY)
# =========================
cap = cv2.VideoCapture(video_input)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 1000, 600)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 BAT DETECTION
    bat_res = bat_model(frame, verbose=False)[0]
    bat_center, bat_box = get_bat_info(bat_res)

    # 🔥 POSE DETECTION
    res = pose_model(frame)[0]

    selected_kps = None
    selected_box = None

    if res.keypoints is not None:
        kps_all = res.keypoints.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()

        best_score = 999999
        for i in range(len(kps_all)):
            score = get_person_score(
                kps_all[i],
                boxes[i],
                bat_center,
                bat_box
            )
            if score < best_score:
                best_score = score
                selected_kps = kps_all[i]
                selected_box = boxes[i]

    if selected_kps is not None:
        draw_pose(frame, selected_kps)

        x1,y1,x2,y2 = map(int, selected_box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        feats = extract_features(selected_kps)

        if feats is not None:
            while True:
                display = frame.copy()

                cv2.putText(display, f"Frame: {frame_id}", (20,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

                cv2.putText(display, "F=Front S=Side W=Wrong Q=Quit",
                            (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                cv2.imshow("Frame", display)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('f'):
                    writer.writerow(feats + ["FRONT"])
                    break
                elif key == ord('s'):
                    writer.writerow(feats + ["SIDE"])
                    break
                elif key == ord('w'):
                    writer.writerow(feats + ["WRONG"])
                    break
                elif key == ord('q'):
                    cap.release()
                    csv_file.close()
                    cv2.destroyAllWindows()
                    exit()

    frame_id += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()