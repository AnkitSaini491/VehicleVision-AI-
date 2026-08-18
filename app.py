
import cv2
import os
from ultralytics import YOLO


# ============================================================
# VEHICLEVISION AI
# Real-Time Vehicle Detection System
# ============================================================


# ============================================================
# VIDEO PATH
# ============================================================

VIDEO_PATH = r"C:\Users\DELL\Downloads\vehicle_video.mp4.mp4"


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_NAME = "yolo11n.pt"


# ============================================================
# VEHICLE CLASSES
# COCO CLASS IDs
# ============================================================

VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_WIDTH = 1000
MAX_HEIGHT = 650


# ============================================================
# CHECK VIDEO
# ============================================================

print("=" * 60)
print("          VEHICLEVISION AI")
print("     REAL-TIME VEHICLE DETECTION")
print("=" * 60)

print()
print("Checking video...")
print("Path:", VIDEO_PATH)

if not os.path.exists(VIDEO_PATH):

    print()
    print("❌ VIDEO NOT FOUND!")
    print()
    print("Make sure the file exists at:")
    print(VIDEO_PATH)
    print()

    input("Press Enter to exit...")
    raise SystemExit

print("✅ VIDEO FOUND!")
print()


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO AI model...")

try:

    model = YOLO(MODEL_NAME)

except Exception as error:

    print("❌ Could not load YOLO model.")
    print(error)

    input("Press Enter to exit...")
    raise SystemExit

print("✅ YOLO MODEL LOADED!")
print()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("❌ Unable to open video.")

    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

video_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

video_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Video Resolution:", video_width, "x", video_height)
print("FPS:", fps)
print("Total Frames:", total_frames)
print()


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "VehicleVision AI - LIVE"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    MAX_WIDTH,
    MAX_HEIGHT
)


# ============================================================
# VIDEO DELAY
# ============================================================

delay = max(
    1,
    int(1000 / fps)
)


# ============================================================
# FRAME COUNTER
# ============================================================

frame_number = 0


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("✅ Video processing completed.")
        break


    frame_number += 1


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    try:

        results = model(
            frame,
            conf=0.40,
            verbose=False
        )

    except Exception as error:

        print("Detection error:", error)
        break


    # ========================================================
    # VEHICLE COUNTERS
    # ========================================================

    total_vehicles = 0

    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )


            # Only vehicle classes
            if class_id not in VEHICLE_CLASSES:
                continue


            total_vehicles += 1


            vehicle_name = VEHICLE_CLASSES[
                class_id
            ]


            # =================================================
            # COUNT VEHICLES
            # =================================================

            if class_id == 2:

                car_count += 1

            elif class_id == 3:

                motorcycle_count += 1

            elif class_id == 5:

                bus_count += 1

            elif class_id == 7:

                truck_count += 1


            # =================================================
            # GET BOUNDING BOX
            # =================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # =================================================
            # GREEN BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # =================================================
            # LABEL
            # =================================================

            label = (
                f"{vehicle_name} "
                f"{confidence * 100:.0f}%"
            )


            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.6

            thickness = 2


            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )


            label_top = max(
                0,
                y1 - text_height - 12
            )


            # Label background

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_top
                ),
                (
                    x1 + text_width + 10,
                    y1
                ),
                (0, 255, 0),
                -1
            )


            # Label text

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    y1 - 6
                ),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )


    # ========================================================
    # HUD BORDER
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (
            video_width - 10,
            video_height - 10
        ),
        (0, 255, 0),
        2
    )


    # ========================================================
    # HEADER BACKGROUND
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (
            video_width - 10,
            65
        ),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "VEHICLEVISION AI",
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # ========================================================
    # TOTAL VEHICLES
    # ========================================================

    cv2.putText(
        frame,
        f"TOTAL VEHICLES : {total_vehicles}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # ========================================================
    # VEHICLE COUNTS
    # ========================================================

    cv2.putText(
        frame,
        f"CARS       : {car_count}",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"MOTORCYCLE : {motorcycle_count}",
        (25, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"BUSES      : {bus_count}",
        (25, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"TRUCKS     : {truck_count}",
        (25, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    # ========================================================
    # AI STATUS PANEL
    # ========================================================

    panel_x = video_width - 310

    cv2.putText(
        frame,
        "AI SYSTEM : ONLINE",
        (
            panel_x,
            105
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "DETECTION : ACTIVE",
        (
            panel_x,
            140
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "CAMERA : ONLINE",
        (
            panel_x,
            175
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    # ========================================================
    # CENTER CROSSHAIR
    # ========================================================

    center_x = video_width // 2
    center_y = video_height // 2


    cv2.circle(
        frame,
        (
            center_x,
            center_y
        ),
        35,
        (0, 255, 0),
        1
    )


    cv2.line(
        frame,
        (
            center_x - 50,
            center_y
        ),
        (
            center_x - 10,
            center_y
        ),
        (0, 255, 0),
        1
    )


    cv2.line(
        frame,
        (
            center_x + 10,
            center_y
        ),
        (
            center_x + 50,
            center_y
        ),
        (0, 255, 0),
        1
    )


    cv2.line(
        frame,
        (
            center_x,
            center_y - 50
        ),
        (
            center_x,
            center_y - 10
        ),
        (0, 255, 0),
        1
    )


    cv2.line(
        frame,
        (
            center_x,
            center_y + 10
        ),
        (
            center_x,
            center_y + 50
        ),
        (0, 255, 0),
        1
    )


    # ========================================================
    # FRAME INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"FRAME : {frame_number}",
        (
            video_width - 220,
            video_height - 55
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # FOOTER
    # ========================================================

    cv2.putText(
        frame,
        "LIVE VEHICLE ANALYSIS",
        (
            25,
            video_height - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Q / ESC = EXIT",
        (
            video_width - 170,
            video_height - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    # ========================================================
    # RESIZE VIDEO FOR SCREEN
    # ========================================================

    scale = min(
        MAX_WIDTH / video_width,
        MAX_HEIGHT / video_height,
        1.0
    )


    display_width = int(
        video_width * scale
    )

    display_height = int(
        video_height * scale
    )


    display_frame = cv2.resize(
        frame,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD CONTROL
    # ========================================================

    key = cv2.waitKey(
        delay
    ) & 0xFF


    # ESC
    if key == 27:
        break


    # Q
    if key == ord("q"):
        break


# ============================================================
# RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 60)
print("✅ VEHICLEVISION AI STOPPED")
print("=" * 60)
