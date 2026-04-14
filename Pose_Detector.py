import cv2
import mediapipe as mp
import numpy as np
import math
import ctypes
from collections import deque

# ===============================
# INITIALIZE MEDIAPIPE
# ===============================

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

POSE_DETECTION_CONFIDENCE = 0.65
POSE_TRACKING_CONFIDENCE = 0.65
HAND_DETECTION_CONFIDENCE = 0.65
HAND_TRACKING_CONFIDENCE = 0.65

pose_static = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=POSE_DETECTION_CONFIDENCE
)
pose_live = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
    smooth_landmarks=True
)

hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=HAND_DETECTION_CONFIDENCE
)
hands_live = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_TRACKING_CONFIDENCE
)


# ===============================
# UTILITY FUNCTIONS
# ===============================

def calculate_angle(a, b, c):
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


# ===============================
# BODY JOINT DEFINITIONS
# ===============================

POSE_JOINTS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}

# Finger joints (for each hand)
FINGER_JOINTS = {
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
    "index_mcp": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}

STATUS_INCREASE = 1
STATUS_DECREASE = -1
STATUS_NO_CHANGE = 0

STATUS_COLORS = {
    STATUS_DECREASE: (0, 0, 255),
    STATUS_INCREASE: (0, 255, 0),
    STATUS_NO_CHANGE: (255, 0, 0),
}

POSE_VISIBILITY_THRESHOLD = 0.35
BODY_ANGLE_VISIBILITY_THRESHOLD = 0.55
BODY_ANGLE_TOLERANCE = 16
FINGER_ANGLE_TOLERANCE = 9

BODY_ANGLE_SMOOTHING_ALPHA = 0.35
HAND_ANGLE_SMOOTHING_ALPHA = 0.4
HAND_MATCH_MAX_MEAN_DIFF = 42
SIMILARITY_HISTORY_SIZE = 10


# ===============================
# EXTRACT REFERENCE POSE
# ===============================

def extract_reference(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose_static.process(image_rgb)
    hands_results = hands_static.process(image_rgb)

    reference = {}

    if pose_results.pose_landmarks:
        body_landmarks = pose_results.pose_landmarks.landmark
        reference["body"] = [(lm.x, lm.y, lm.z, lm.visibility) for lm in body_landmarks]
        reference["body_angles"] = compute_body_angles(
            reference["body"],
            visibility_threshold=BODY_ANGLE_VISIBILITY_THRESHOLD
        )

    if hands_results.multi_hand_landmarks:
        reference["hands"] = []
        reference["hand_angles"] = []
        reference["hand_angles_by_label"] = {}

        for i, hand in enumerate(hands_results.multi_hand_landmarks):
            hand_lm = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            reference["hands"].append(hand_lm)
            hand_angles = compute_hand_angles(hand_lm)
            reference["hand_angles"].append(hand_angles)

            hand_label = None
            if hands_results.multi_handedness and i < len(hands_results.multi_handedness):
                hand_label = hands_results.multi_handedness[i].classification[0].label.lower()
            if hand_label:
                reference["hand_angles_by_label"][hand_label] = hand_angles

    reference["reference_image"] = image
    return reference


# ===============================
# ANGLE COMPUTATION
# ===============================

def compute_body_angles(landmarks, visibility_threshold=None):
    angles = {}
    for name, (a, b, c) in POSE_JOINTS.items():
        if max(a, b, c) >= len(landmarks):
            continue

        if visibility_threshold is not None:
            joint_points = (landmarks[a], landmarks[b], landmarks[c])
            if any(len(pt) < 4 or pt[3] < visibility_threshold for pt in joint_points):
                continue

        angles[name] = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
    return angles


def compute_hand_angles(landmarks):
    angles = {}
    for name, (a, b, c) in FINGER_JOINTS.items():
        angles[name] = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
    return angles


def smooth_angle_dict(previous, current, alpha):
    if not previous:
        return current.copy()

    smoothed = {}
    for joint, angle in current.items():
        prev_angle = previous.get(joint, angle)
        smoothed[joint] = (alpha * angle) + ((1 - alpha) * prev_angle)
    return smoothed


def mean_abs_joint_diff(reference_angles, live_angles):
    common_joints = [joint for joint in reference_angles if joint in live_angles]
    if not common_joints:
        return float("inf")

    diff_sum = sum(abs(live_angles[joint] - reference_angles[joint]) for joint in common_joints)
    return diff_sum / len(common_joints)


def match_live_hands_to_reference(live_hand_angles, reference_hand_angles):
    assignments = {}
    unmatched_reference = set(range(len(reference_hand_angles)))

    for live_idx, live_angles in enumerate(live_hand_angles):
        if not unmatched_reference:
            break

        best_ref = None
        best_diff = float("inf")
        for ref_idx in unmatched_reference:
            diff = mean_abs_joint_diff(reference_hand_angles[ref_idx], live_angles)
            if diff < best_diff:
                best_diff = diff
                best_ref = ref_idx

        if best_ref is not None and best_diff <= HAND_MATCH_MAX_MEAN_DIFF:
            assignments[live_idx] = best_ref
            unmatched_reference.remove(best_ref)

    return assignments


# ===============================
# FEEDBACK GENERATOR
# ===============================

def generate_feedback(
    ref_angles,
    live_angles,
    threshold_body=BODY_ANGLE_TOLERANCE,
    threshold_finger=FINGER_ANGLE_TOLERANCE
):
    feedback = []
    score = 0.0
    compared_count = 0
    joint_status = {}

    for joint in ref_angles:
        if joint not in live_angles:
            continue

        compared_count += 1

        diff = live_angles[joint] - ref_angles[joint]
        abs_diff = abs(diff)

        threshold = threshold_finger if "mcp" in joint or "dip" in joint else threshold_body

        per_joint_score = max(0.0, 1.0 - (abs_diff / (threshold * 2)))
        score += per_joint_score

        if abs_diff <= threshold:
            joint_status[joint] = STATUS_NO_CHANGE
        else:
            if diff > 0:
                feedback.append((joint, STATUS_DECREASE, abs_diff))
                joint_status[joint] = STATUS_DECREASE
            else:
                feedback.append((joint, STATUS_INCREASE, abs_diff))
                joint_status[joint] = STATUS_INCREASE

    total_reference_joints = max(1, len(ref_angles))
    if compared_count == 0:
        similarity = 0
    else:
        coverage = compared_count / total_reference_joints
        similarity = int(((score / compared_count) * coverage) * 100)
    return feedback, similarity, joint_status


def format_joint_name(joint_name):
    return joint_name.replace("_", " ").title()


def format_action_label(status):
    if status == STATUS_DECREASE:
        return "Decrease angle"
    if status == STATUS_INCREASE:
        return "Increase angle"
    return "Hold"


def draw_feedback_panel(frame, feedback_items, max_items=6):
    if not feedback_items:
        cv2.rectangle(frame, (16, 18), (380, 58), (35, 35, 35), -1)
        cv2.rectangle(frame, (16, 18), (380, 58), (255, 255, 255), 1)
        cv2.putText(
            frame,
            "Great alignment. Keep holding.",
            (28, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        return

    feedback_items = sorted(feedback_items, key=lambda item: item[2], reverse=True)

    start_x = 16
    start_y = 18
    card_w = 500
    card_h = 34
    gap = 8

    for i, (joint_name, status, abs_diff) in enumerate(feedback_items[:max_items]):
        top = start_y + i * (card_h + gap)
        bottom = top + card_h

        cv2.rectangle(frame, (start_x, top), (start_x + card_w, bottom), (35, 35, 35), -1)
        cv2.rectangle(frame, (start_x, top), (start_x + card_w, bottom), (255, 255, 255), 1)

        color = STATUS_COLORS.get(status, STATUS_COLORS[STATUS_NO_CHANGE])
        cv2.rectangle(frame, (start_x + 6, top + 6), (start_x + 24, bottom - 6), color, -1)

        label = f"{format_joint_name(joint_name)}: {format_action_label(status)} ({int(abs_diff)}deg)"
        cv2.putText(
            frame,
            label,
            (start_x + 32, top + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )


def draw_joint_feedback_dots(frame, landmarks, joint_definitions, status_by_joint, radius=8, visibility_threshold=None):
    frame_h, frame_w = frame.shape[:2]

    for joint_name, (_, b, _) in joint_definitions.items():
        if b >= len(landmarks):
            continue

        if visibility_threshold is not None and len(landmarks[b]) >= 4:
            if landmarks[b][3] < visibility_threshold:
                continue

        status = status_by_joint.get(joint_name, STATUS_NO_CHANGE)
        color = STATUS_COLORS.get(status, STATUS_COLORS[STATUS_NO_CHANGE])

        x = int(landmarks[b][0] * frame_w)
        y = int(landmarks[b][1] * frame_h)

        if 0 <= x < frame_w and 0 <= y < frame_h:
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), radius, color, -1)


def draw_reference_detections(reference_image, reference):
    if reference_image is None:
        return None

    annotated = reference_image.copy()
    frame_h, frame_w = annotated.shape[:2]

    if "body" in reference:
        body_lm = reference["body"]
        for a, b in mp_pose.POSE_CONNECTIONS:
            if a < len(body_lm) and b < len(body_lm):
                x1, y1 = int(body_lm[a][0] * frame_w), int(body_lm[a][1] * frame_h)
                x2, y2 = int(body_lm[b][0] * frame_w), int(body_lm[b][1] * frame_h)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for x, y, *_ in body_lm:
            px, py = int(x * frame_w), int(y * frame_h)
            if 0 <= px < frame_w and 0 <= py < frame_h:
                cv2.circle(annotated, (px, py), 3, (255, 255, 255), -1)

        draw_joint_feedback_dots(annotated, body_lm, POSE_JOINTS, {}, radius=8)

    if "hands" in reference:
        for hand_lm in reference["hands"]:
            for a, b in mp_hands.HAND_CONNECTIONS:
                if a < len(hand_lm) and b < len(hand_lm):
                    x1, y1 = int(hand_lm[a][0] * frame_w), int(hand_lm[a][1] * frame_h)
                    x2, y2 = int(hand_lm[b][0] * frame_w), int(hand_lm[b][1] * frame_h)
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)

            for x, y, _ in hand_lm:
                px, py = int(x * frame_w), int(y * frame_h)
                if 0 <= px < frame_w and 0 <= py < frame_h:
                    cv2.circle(annotated, (px, py), 3, (255, 255, 255), -1)

            draw_joint_feedback_dots(annotated, hand_lm, FINGER_JOINTS, {}, radius=6)

    return annotated


def prepare_reference_panel(reference, target_height):
    reference_image = reference.get("reference_annotated_image")
    if reference_image is None:
        reference_image = reference.get("reference_image")
    if reference_image is None:
        return None

    ref_h, ref_w = reference_image.shape[:2]
    if ref_h == 0 or ref_w == 0:
        return None

    new_w = int((target_height / ref_h) * ref_w)
    resized = cv2.resize(reference_image, (new_w, target_height))
    cv2.putText(
        resized,
        "Reference",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return resized


def draw_color_legend(frame):
    legend_colors = [
        STATUS_COLORS[STATUS_DECREASE],
        STATUS_COLORS[STATUS_INCREASE],
        STATUS_COLORS[STATUS_NO_CHANGE],
    ]

    box_w = 255
    box_h = 102
    start_x = max(10, frame.shape[1] - box_w - 10)
    start_y = 10

    cv2.rectangle(frame, (start_x, start_y), (start_x + box_w, start_y + box_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + box_w, start_y + box_h), (255, 255, 255), 1)
    cv2.putText(frame, "Legend", (start_x + 10, start_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    labels = ["Decrease angle", "Increase angle", "Aligned"]
    y = start_y + 43
    for i, color in enumerate(legend_colors):
        cv2.circle(frame, (start_x + 16, y), 9, (255, 255, 255), -1)
        cv2.circle(frame, (start_x + 16, y), 7, color, -1)
        cv2.putText(frame, labels[i], (start_x + 32, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 24


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def fit_frame_to_screen(frame, max_width, max_height):
    frame_h, frame_w = frame.shape[:2]
    if frame_h == 0 or frame_w == 0:
        return frame

    scale = min(max_width / frame_w, max_height / frame_h, 1.0)
    if scale >= 1.0:
        return frame

    new_w = max(1, int(frame_w * scale))
    new_h = max(1, int(frame_h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ===============================
# LIVE TRACKING
# ===============================

def start_live(reference):
    window_name = "Pose Replication Coach"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)





    
    screen_w, screen_h = get_screen_size()
    max_window_w = int(screen_w * 0.92)
    max_window_h = int(screen_h * 0.88)

    if "reference_image" in reference and "reference_annotated_image" not in reference:
        reference["reference_annotated_image"] = draw_reference_detections(
            reference.get("reference_image"),
            reference
        )

    smoothed_body_angles = {}
    smoothed_hand_angles_by_ref = {}
    similarity_history = deque(maxlen=SIMILARITY_HISTORY_SIZE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = pose_live.process(image_rgb)
        hand_result = hands_live.process(image_rgb)

        all_feedback = []
        similarity_scores = []

        # ---- BODY ----
        if pose_result.pose_landmarks and "body_angles" in reference:
            body_lm = [(lm.x, lm.y, lm.z, lm.visibility) for lm in pose_result.pose_landmarks.landmark]
            live_body_angles_raw = compute_body_angles(
                body_lm,
                visibility_threshold=BODY_ANGLE_VISIBILITY_THRESHOLD
            )
            live_body_angles = smooth_angle_dict(
                smoothed_body_angles,
                live_body_angles_raw,
                BODY_ANGLE_SMOOTHING_ALPHA,
            )
            smoothed_body_angles = live_body_angles

            feedback, similarity, body_status = generate_feedback(reference["body_angles"], live_body_angles)
            all_feedback.extend(feedback)
            similarity_scores.append(similarity)

            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_joint_feedback_dots(
                frame,
                body_lm,
                POSE_JOINTS,
                body_status,
                radius=9,
                visibility_threshold=POSE_VISIBILITY_THRESHOLD
            )
        elif "body_angles" in reference:
            cv2.putText(frame, "Body not detected", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---- HANDS ----
        if hand_result.multi_hand_landmarks and "hand_angles" in reference:

            live_hand_data = []
            for hand_landmarks in hand_result.multi_hand_landmarks:
                hand_lm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                live_hand_data.append(
                    {
                        "landmarks": hand_lm,
                        "angles": compute_hand_angles(hand_lm),
                        "status": {},
                    }
                )

            live_hand_angles_list = [item["angles"] for item in live_hand_data]
            hand_assignments = match_live_hands_to_reference(live_hand_angles_list, reference["hand_angles"])

            matched_ref_indices = set()
            for live_idx, ref_idx in hand_assignments.items():
                if ref_idx in matched_ref_indices:
                    continue

                matched_ref_indices.add(ref_idx)
                ref_hand_angles = reference["hand_angles"][ref_idx]
                live_angles = live_hand_data[live_idx]["angles"]
                smoothed_live_angles = smooth_angle_dict(
                    smoothed_hand_angles_by_ref.get(ref_idx, {}),
                    live_angles,
                    HAND_ANGLE_SMOOTHING_ALPHA,
                )
                smoothed_hand_angles_by_ref[ref_idx] = smoothed_live_angles

                feedback, similarity, hand_status = generate_feedback(ref_hand_angles, smoothed_live_angles)
                all_feedback.extend(feedback)
                similarity_scores.append(similarity)
                live_hand_data[live_idx]["status"] = hand_status

            for i, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw_joint_feedback_dots(
                    frame,
                    live_hand_data[i]["landmarks"],
                    FINGER_JOINTS,
                    live_hand_data[i]["status"],
                    radius=7,
                )
        elif "hand_angles" in reference:
            cv2.putText(frame, "Hand not detected", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # ---- DISPLAY FEEDBACK ----
        draw_feedback_panel(frame, all_feedback)

        # ---- DISPLAY SIMILARITY ----
        if similarity_scores:
            avg_similarity = sum(similarity_scores) // len(similarity_scores)
            similarity_history.append(avg_similarity)
            stable_similarity = int(sum(similarity_history) / len(similarity_history))
            score_color = (0, 255, 0) if stable_similarity >= 80 else (0, 220, 255) if stable_similarity >= 60 else (0, 0, 255)
            cv2.putText(frame, f"Similarity: {stable_similarity}%",
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        score_color, 3)

            coaching_text = "Excellent hold" if stable_similarity >= 85 else "Nearly there" if stable_similarity >= 65 else "Adjust highlighted joints"
            cv2.putText(
                frame,
                coaching_text,
                (20, 490),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                score_color,
                2,
            )

        draw_color_legend(frame)

        reference_panel = prepare_reference_panel(reference, frame.shape[0])
        if reference_panel is not None:
            combined = np.hstack((reference_panel, frame))
            display_frame = fit_frame_to_screen(combined, max_window_w, max_window_h)
            cv2.imshow(window_name, display_frame)
        else:
            display_frame = fit_frame_to_screen(frame, max_window_w, max_window_h)
            cv2.imshow(window_name, display_frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    reference_image_path = "reference1.jpg"  # Put your image here

    reference_data = extract_reference(reference_image_path)

    if not reference_data:
        print("Could not detect pose in reference image.")
    else:
        print("Reference pose loaded successfully!")
        start_live(reference_data)