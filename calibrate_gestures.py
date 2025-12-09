# calibrate_gestures.py
import cv2, time, os, numpy as np, mediapipe as mp
from collections import defaultdict

# tiny copy of the analyzers used in main.py
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.4, min_tracking_confidence=0.4)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.4, min_tracking_confidence=0.4)

def lm_to_point(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, getattr(landmark, "z", 0.0)])

def analyze_frame(face_lms, hand_results, w, h):
    res = {}
    if not face_lms:
        return None
    lm = face_lms[0]
    # face landmarks indices used
    ids = {"nose":1, "forehead":10, "left_eye_outer":33, "right_eye_outer":263,
           "left_eye_top":159, "left_eye_bottom":145, "right_eye_top":386, "right_eye_bottom":374,
           "mouth_l":61, "mouth_r":291, "upper_lip":13, "lower_lip":14}
    def p(i): return lm_to_point(lm.landmark[i], w, h)
    try:
        nose = p(ids["nose"])[:2]; forehead = p(ids["forehead"])[:2]
        left_eye_outer = p(ids["left_eye_outer"])[:2]; right_eye_outer = p(ids["right_eye_outer"])[:2]
        le_top = p(ids["left_eye_top"])[:2]; le_bot = p(ids["left_eye_bottom"])[:2]
        re_top = p(ids["right_eye_top"])[:2]; re_bot = p(ids["right_eye_bottom"])[:2]
        mouth_l = p(ids["mouth_l"])[:2]; mouth_r = p(ids["mouth_r"])[:2]
        upper = p(ids["upper_lip"])[:2]; lower = p(ids["lower_lip"])[:2]
    except Exception:
        return None

    eye_dist = np.linalg.norm(left_eye_outer - right_eye_outer) + 1e-6
    mouth_open_rel = np.linalg.norm(lower - upper) / eye_dist
    mouth_center = (mouth_l + mouth_r) / 2.0
    lower_vs_center = (lower[1] - mouth_center[1]) / eye_dist
    left_eye_open = np.linalg.norm(le_top - le_bot) / eye_dist
    right_eye_open = np.linalg.norm(re_top - re_bot) / eye_dist
    eye_open = (left_eye_open + right_eye_open) / 2.0

    res.update({"nose": nose, "forehead": forehead, "eye_dist": eye_dist,
                "mouth_open_rel": mouth_open_rel, "lower_vs_center": lower_vs_center,
                "eye_open": eye_open, "mouth_center": mouth_center})

    # hands: pick first hand if present
    if hand_results and hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z] for p in hand.landmark])
        idx_tip = pts[8][:2]; idx_mcp = pts[5][:2]; wrist = pts[0][:2]; palm = np.mean(pts[[0,1,5,9,13],:2], axis=0)
        tip_to_mcp = np.linalg.norm(idx_tip - idx_mcp) + 1e-6
        mcp_to_wrist = np.linalg.norm(idx_mcp - wrist) + 1e-6
        ext_ratio = tip_to_mcp / mcp_to_wrist
        res.update({
            "index_tip": idx_tip, "index_mcp": idx_mcp, "wrist": wrist, "palm": palm,
            "dist_idx_forehead": np.linalg.norm(idx_tip - forehead) / eye_dist,
            "dist_wrist_forehead": np.linalg.norm(wrist - forehead) / eye_dist,
            "idx_y_rel": (idx_tip[1] - nose[1]) / eye_dist,
            "lateral": abs(idx_tip[0] - nose[0]) / eye_dist,
            "index_ext_ratio": ext_ratio
        })
    return res

def median(vals): 
    arr = np.array(vals)
    return float(np.median(arr)) if len(arr)>0 else None

def run():
    print("Calibration tool\nPress:")
    print("  r - record 'hand_raise' (raise index above nose),")
    print("  t - record 'think' (touch temple with index),")
    print("  o - record 'ohno' (wrist/palm on top of head),")
    print("  q - finish and compute recommended thresholds\n")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    groups = defaultdict(list)
    collecting = None
    collect_until = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(rgb)
            hand_res = hands.process(rgb)
            info = analyze_frame(face_res.multi_face_landmarks if face_res else None, hand_res, w, h)
            # overlay small markers
            if info:
                nx, ny = int(info["nose"][0]), int(info["nose"][1])
                fx, fy = int(info["forehead"][0]), int(info["forehead"][1])
                cv2.circle(frame, (nx, ny), 3, (0,128,255), -1)
                cv2.circle(frame, (fx, fy), 3, (255,0,255), -1)
                if "index_tip" in info:
                    ix, iy = int(info["index_tip"][0]), int(info["index_tip"][1])
                    cv2.circle(frame, (ix, iy), 5, (0,200,0), 2)
            cv2.putText(frame, f"Recording: {collecting or 'none'}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
            cv2.imshow("calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('r'), ord('t'), ord('o')):
                # start recording for next 1.5 seconds
                if key == ord('r'): collecting = 'hand_raise'
                if key == ord('t'): collecting = 'think'
                if key == ord('o'): collecting = 'ohno'
                collect_until = time.time() + 1.5
                print(f"Collecting samples for {collecting} ... hold the pose now")
            # if currently collecting and have info, store it
            if collecting and time.time() < collect_until and info:
                groups[collecting].append(info)
            if collecting and time.time() >= collect_until:
                print(f"Done collecting {collecting}, samples: {len(groups[collecting])}")
                collecting = None
        # finished -> compute medians and suggest thresholds
        print("\nCalibration complete. Computing medians...")
        suggestions = {}
        # hand_raise relevant: idx_y_rel (should be negative), index_ext_ratio, dist_idx_forehead
        hrs = groups['hand_raise']
        if hrs:
            ys = [g['idx_y_rel'] for g in hrs if 'idx_y_rel' in g]
            ex = [g['index_ext_ratio'] for g in hrs if 'index_ext_ratio' in g]
            dfore = [g['dist_idx_forehead'] for g in hrs if 'dist_idx_forehead' in g]
            suggestions['HAND_RAISE_REL_Y'] = median(ys) - 0.03  # a bit stricter than your median
            suggestions['INDEX_EXTENDED_RATIO'] = median(ex) - 0.08
            suggestions['DIST_IDX_FOREHEAD_FOR_RAISE'] = median(dfore)
        ths = groups['think']
        if ths:
            dfore = [g['dist_idx_forehead'] for g in ths if 'dist_idx_forehead' in g]
            lateral = [g['lateral'] for g in ths if 'lateral' in g]
            suggestions['TOUCH_HEAD_DIST'] = median(dfore) + 0.02
            suggestions['THINK_LATERAL'] = median(lateral) - 0.05
        ohs = groups['ohno']
        if ohs:
            dw = [g['dist_wrist_forehead'] for g in ohs if 'dist_wrist_forehead' in g]
            suggestions['HAND_ON_TOP_DIST'] = median(dw) + 0.03
        print("\nSamples collected:")
        for k,v in ((k,len(groups[k])) for k in groups):
            print(f"  {k}: {v} samples")
        print("\nSuggested thresholds (paste into main.py):")
        for k,v in suggestions.items():
            print(f"{k} = {v:.4f}")
        if not suggestions:
            print("No samples collected — try again and be sure to press r/t/o while holding the pose.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run()

