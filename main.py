# main.py — simple version (tongue, eyes closed, hand gestures)
import cv2
import mediapipe as mp
import numpy as np
import imageio
import os
import time

# ---------- CONFIG ----------
ASSETS_DIR = os.path.expanduser("~/freak/gifs")
REACTION_GIFS = {
    "tongue": "catkitty.gif",            # tongue out
    "eyes_closed": "crackingup.gif",     # eyes closed
    "hand_raise": "instagrammonkey.gif", # index finger up
    "think": "think.gif",                # finger to head
    "ohno": "ohmg.gif",                  # hand on head
    "neutral": None
}

# basic thresholds
MOUTH_OPEN_TONGUE = 0.30
TONGUE_PROTRUDE_DIFF = 0.020
EYE_CLOSED_THRESH = 0.06
HAND_ABOVE_FACE_Y_GAP = -0.05
TOUCH_HEAD_DIST = 0.12
HAND_ON_TOP_DIST = 0.13
INDEX_EXTENDED_RATIO = 0.6
# -----------------------------

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ---- GIF loading ----
def load_gif_frames(path):
    gif = imageio.mimread(path)
    frames = []
    for im in gif:
        arr = np.array(im).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        frames.append(bgr)
    return frames

gif_cache = {}
for name, fname in REACTION_GIFS.items():
    if fname is None:
        continue
    path = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(path):
        try:
            frames = load_gif_frames(path)
            gif_cache[name] = frames
            print(f"Loaded GIF {fname} for '{name}' ({len(frames)} frames)")
        except Exception as e:
            print("Error loading", path, e)
    else:
        print("Missing GIF:", path)

# ---- helpers ----
def lm_to_point(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])

def analyze_face_landmarks(landmarks, w, h):
    idx = {
        "mouth_l": 61, "mouth_r": 291, "upper_lip": 13, "lower_lip": 14,
        "left_eye_top": 159, "left_eye_bottom": 145,
        "right_eye_top": 386, "right_eye_bottom": 374,
        "left_eye_outer": 33, "right_eye_outer": 263,
        "nose_tip": 1, "forehead_ref": 10
    }
    def p(i): return lm_to_point(landmarks[i], w, h)
    try:
        mouth_l = p(idx["mouth_l"]); mouth_r = p(idx["mouth_r"])
        upper = p(idx["upper_lip"]); lower = p(idx["lower_lip"])
        le_top = p(idx["left_eye_top"]); le_bot = p(idx["left_eye_bottom"])
        re_top = p(idx["right_eye_top"]); re_bot = p(idx["right_eye_bottom"])
        left_eye_outer = p(idx["left_eye_outer"]); right_eye_outer = p(idx["right_eye_outer"])
        nose = p(idx["nose_tip"]); forehead = p(idx["forehead_ref"])
    except Exception:
        return None

    eye_dist = np.linalg.norm(left_eye_outer - right_eye_outer) + 1e-6
    mouth_open = np.linalg.norm(lower - upper)
    mouth_open_rel = mouth_open / eye_dist
    mouth_center = (mouth_l + mouth_r) / 2.0

    left_eye_open = np.linalg.norm(le_top - le_bot) / eye_dist
    right_eye_open = np.linalg.norm(re_top - re_bot) / eye_dist
    eye_open = (left_eye_open + right_eye_open) / 2.0

    lower_vs_center = (lower[1] - mouth_center[1]) / eye_dist

    return {
        "eye_open": eye_open,
        "mouth_open_rel": mouth_open_rel,
        "lower_vs_center": lower_vs_center,
        "nose": nose,
        "forehead": forehead,
        "eye_dist": eye_dist,
        "mouth_center": mouth_center
    }

def analyze_hands(hand_results, w, h):
    hands_list = []
    if not hand_results.multi_hand_landmarks:
        return hands_list
    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                          hand_results.multi_handedness):
        lm = hand_landmarks.landmark
        pts = np.array([[p.x * w, p.y * h, p.z] for p in lm])
        index_tip = pts[8][:2]
        index_mcp = pts[5][:2]
        wrist = pts[0][:2]
        palm = np.mean(pts[[0,1,5,9,13], :2], axis=0)
        hands_list.append({
            "index_tip": index_tip,
            "index_mcp": index_mcp,
            "wrist": wrist,
            "palm": palm,
            "landmarks": pts,
            "label": handedness.classification[0].label
        })
    return hands_list

def is_index_extended(h):
    tip = h["index_tip"]; mcp = h["index_mcp"]; wrist = h["wrist"]
    tip_to_mcp = np.linalg.norm(tip - mcp) + 1e-6
    mcp_to_wrist = np.linalg.norm(mcp - wrist) + 1e-6
    ratio = tip_to_mcp / mcp_to_wrist
    return ratio > INDEX_EXTENDED_RATIO

def choose_reaction(face_data, hands_list):
    if face_data is None:
        return "neutral"

    nose = face_data["nose"]
    forehead = face_data["forehead"]
    eye_dist = face_data["eye_dist"]

    # hand-based reactions
    for h in hands_list:
        idx = h["index_tip"]
        wrist = h["wrist"]
        palm = h["palm"]

        dist_idx_forehead = np.linalg.norm(idx - forehead) / eye_dist
        dist_wrist_forehead = np.linalg.norm(wrist - forehead) / eye_dist
        rel_idx_vs_nose = (idx[1] - nose[1]) / eye_dist   # negative => above
        lateral = abs(idx[0] - nose[0]) / eye_dist

        # think: index near side of head (temple-ish)
        if dist_idx_forehead < TOUCH_HEAD_DIST and lateral > 0.5:
            return "think"

        # ohno: wrist or palm near forehead
        if dist_wrist_forehead < HAND_ON_TOP_DIST or np.linalg.norm(palm - forehead) / eye_dist < HAND_ON_TOP_DIST:
            return "ohno"

        # hand raise: index above nose & extended & not touching head
        if rel_idx_vs_nose < HAND_ABOVE_FACE_Y_GAP and is_index_extended(h) and dist_idx_forehead > TOUCH_HEAD_DIST:
            return "hand_raise"

    # face-only
    if face_data["eye_open"] < EYE_CLOSED_THRESH:
        return "eyes_closed"

    if face_data["mouth_open_rel"] > MOUTH_OPEN_TONGUE and face_data["lower_vs_center"] > TONGUE_PROTRUDE_DIFF:
        return "tongue"

    return "neutral"

# ---- main loop ----
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    last_reaction = None
    last_change_time = time.time()
    forced = None

    print("Controls: q=quit, 1..5 force reactions, 0=auto")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        face_data = None
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            face_data = analyze_face_landmarks(lm, w, h)

        hands_list = analyze_hands(hand_results, w, h)

        candidate = choose_reaction(face_data, hands_list)

        if forced is not None:
            candidate = forced

        reaction = candidate

        if reaction != last_reaction:
            last_reaction = reaction
            last_change_time = time.time()

        status = f"reaction:{reaction}"
        cv2.putText(frame, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("webcam", frame)

        # show gif
        if reaction in gif_cache:
            t_since = time.time() - last_change_time
            frames = gif_cache[reaction]
            idx = int((t_since * 10) % len(frames))
            frm = frames[idx]
            frm_small = cv2.resize(frm, (320, int(320 * frm.shape[0] / frm.shape[1])))
            cv2.imshow("reaction", frm_small)
        else:
            cv2.imshow("reaction", np.zeros((240,320,3), dtype=np.uint8))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            forced = "tongue"
        elif key == ord('2'):
            forced = "eyes_closed"
        elif key == ord('3'):
            forced = "hand_raise"
        elif key == ord('4'):
            forced = "think"
        elif key == ord('5'):
            forced = "ohno"
        elif key == ord('0'):
            forced = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
