# Importing libraries
import os
import platform
import subprocess
import platform
import subprocess
import time
import cv2 # webcam during testing
from threading import Thread # audio helper
from ultralytics import YOLO # yolo for object detection
from playsound3 import playsound # audio

# for tts
try:
    import pyttsx3 
except ImportError:
    pyttsx3 = None

# for reading data from ultrasonic sensor
try:
    import serial
except ImportError:
    serial = None

# for ai descriptions
try:
    import openai
except ImportError:
    openai = None

# Config stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(script_dir, "[Pigsy]What kind of object is this.mp3") # example sound 
window_name = "camera"

# Model configuration (using latest version of YOLO)
model_path = "yolo26n.pt"
if not os.path.isfile(model_path):
    alt_model = "yolov8n.pt"
    if os.path.isfile(alt_model):
        model_path = alt_model

# yolo model used, can be changed out if needed.
model = YOLO(model_path)
model.conf = 0.25
model.iou = 0.45

# Ultrasonic sensor configuration
ultrasonic_enabled = True
ultrasonic_port_name = "COM3"
ultrasonic_baud = 9600

# Description configuration
enable_ai_description = True
confidence_threshold = 0.25

# Distance tracker configuration
focal_length = 800.0  # adjust based on camera calibration
known_object_heights = { # note: adjust + add more of these in the future
    "person": 1.7,
    "bicycle": 1.2,
    "car": 1.5,
    "motorbike": 1.4,
    "bus": 3.0,
    "truck": 3.0,
    "chair": 1.0,
    "sofa": 1.0,
    "dog": 0.6,
    "cat": 0.3,
    "table": 0.8,
    "bench": 0.8,
    "bed": 1.2,
}

# list of road hazards
hazard_collision_labels = {
    "car",
    "truck",
    "bus",
    "motorbike",
    "bicycle",
    "person",
}

#list of household hazards
hazard_household_labels = {
    "knife",
    "scissors",
    "cup",
    "bottle",
    "kettle",
    "oven",
    "microwave",
    "stove",
    "chair",
    "sofa",
    "couch",
    "table",
    "bed",
    "stairs",
    "door",
    "book",
}

# Camera setup section,
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Cannot open camera")
    capture.release()
    cv2.destroyAllWindows()
    raise SystemExit

# Set desired camera resolution
desired_width = 1280
desired_height = 720
capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened at resolution: {actual_width}x{actual_height}")

# used for debugging on windows device, likely taken out when deployed on the actual glasses
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, actual_width, actual_height)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
print("Camera opened. Click on the camera window and press any key to start.")

# Wait for initial key press before starting processing
while True:
    ret, frame = capture.read()
    if not ret or frame is None:
        print("Error: Failed to read frame from camera")
        capture.release()
        cv2.destroyAllWindows()
        raise SystemExit

    cv2.imshow(window_name, frame)
    start_key = cv2.waitKey(10) & 0xFF
    if start_key != 255 and start_key != 0:
        print(f"Starting camera after input key: {start_key}")
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed before start, quitting...")
        capture.release()
        cv2.destroyAllWindows()
        raise SystemExit

# Audio section
sound_playing = False

# Validate the sound file before use
def valid_sound_file(path):
    return path and os.path.isfile(path)

if not valid_sound_file(sound_path):
    # debug
    print("Warning: sound file not found. Audio playback is disabled.")
    print("Expected sound file at:", sound_path)
    sound_path = None

engine = None
ultrasonic_serial = None

# input debouncing 
key_debounce_ms = 250
last_key = None
last_key_time = 0
last_detected_objects = []

def estimate_distance_text(label, box, frame_height):
    pixel_height = box[3] - box[1]
    if pixel_height <= 0:
        return None

    known_height = known_object_heights.get(label.lower())
    if known_height:
        meters = (known_height * focal_length) / pixel_height
        return f"{meters:.1f} m"

    ratio = frame_height / pixel_height
    if ratio > 6:
        return "Far"
    if ratio > 3:
        return "Medium"
    return "Close"

# function to help convert raw images into results
def process_detection_results(results, frame_height):
    objects = []
    if not results or len(results) == 0:
        return objects

    r = results[0]
    names = getattr(r, "names", {}) or {}
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return objects

    cls_values = getattr(boxes, "cls", None)
    conf_values = getattr(boxes, "conf", None)
    xyxy_values = getattr(boxes, "xyxy", None)
    if cls_values is None or xyxy_values is None:
        return objects

    try:
        cls_list = cls_values.cpu().numpy().tolist()
    except Exception:
        cls_list = list(cls_values)

    try:
        xyxy_list = xyxy_values.cpu().numpy().tolist()
    except Exception:
        xyxy_list = list(xyxy_values)

    conf_list = []
    if conf_values is not None:
        try:
            conf_list = conf_values.cpu().numpy().tolist()
        except Exception:
            conf_list = list(conf_values)
    
    # converts raw predictrons from YOLO and converts it to structured list of detected objects
    for idx, cls_id in enumerate(cls_list):
        label = names.get(int(cls_id), str(int(cls_id)))
        confidence = conf_list[idx] if idx < len(conf_list) else None
        box = xyxy_list[idx]
        distance = estimate_distance_text(label, box, frame_height)
        display_label = f"{label}"
        if confidence is not None:
            display_label += f" ({confidence:.2f})"
        if distance is not None:
            display_label += f" [{distance}]"
        objects.append({
            "label": label,
            "confidence": confidence,
            "distance": distance,
            "display": display_label,
            "box": box,
        })

    return objects

# For tts functionality 
def init_tts():
    if not pyttsx3:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        return engine
    except Exception as e:
        print("Warning: TTS initialization failed:", e)
        return None

# if tts fails, this is a fall back
def speak_windows_fallback(text):
    try:
        powershell_command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([Console]::In.ReadToEnd())"
        ]
        proc = subprocess.Popen(
            powershell_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        proc.communicate(text)
    except Exception as e:
        print("Warning: Windows TTS fallback failed:", e)
        print(text)

# ensures tts can simultaneously occur while video occurs.
def speak(text):
    if not text:
        return

    def _speak():
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("Warning: text-to-speech failed:", e)
                if platform.system() == "Windows":
                    speak_windows_fallback(text)
                else:
                    print(text)
        elif platform.system() == "Windows":
            speak_windows_fallback(text)
        else:
            print("TTS unavailable. Message:", text)

    Thread(target=_speak, daemon=True).start()

# function to interact with ultrasonic sensor component
def init_ultrasonic_sensor():
    if not serial or not ultrasonic_enabled:
        return None
    try:
        ser = serial.Serial(ultrasonic_port_name, ultrasonic_baud, timeout=0.1)
        print(f"Ultrasonic sensor connected on {ultrasonic_port_name}")
        return ser
    except Exception as e:
        print(f"Ultrasonic sensor unavailable: {e}")
        return None

# function to format raw data to 
def read_ultrasonic_distance(serial_port):
    if serial_port is None:
        return None
    try:
        line = serial_port.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        value = float(line)
        return value
    except Exception:
        return None

# function to help determine and objects pos in frame
def object_position(box, frame_width):
    center_x = (box[0] + box[2]) / 2
    if center_x < frame_width * 0.33:
        return "left"
    if center_x > frame_width * 0.66:
        return "right"
    return "ahead"

# building the open ai prompt to report back to the user
def build_scene_prompt(objects, ultrasonic_distance, frame_width):
    prompt_lines = [
        "You are an assistive AI describing the scene to a visually impaired user.",
        "Use short, clear statements and mention hazards, distances, positions, and avoidance guidance.",
        "Only output up to three concise messages separated by new lines.",
        "Detected objects:",
    ]
    for obj in objects:
        label = obj["label"].lower()
        position = object_position(obj["box"], frame_width)
        distance = obj["distance"] or "unknown distance"
        prompt_lines.append(f"- {label} at {distance} on the {position}")

    if ultrasonic_distance is not None:
        prompt_lines.append(f"Ultrasonic sensor distance: {ultrasonic_distance:.1f} meters.")

    prompt_lines.append("Generate the spoken guidance now.")
    return "\n".join(prompt_lines)

# using the earlier prompt to generate
def generate_ai_description(objects, ultrasonic_distance, frame_width):
    if not enable_ai_description or not openai or not os.getenv("OPENAI_API_KEY"):
        return None
    prompt = build_scene_prompt(objects, ultrasonic_distance, frame_width)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You provide assistive guidance for visually impaired users."},
                      {"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.6,
        )
        content = response.choices[0].message.content.strip()
        if not content:
            return None
        return [line.strip() for line in content.splitlines() if line.strip()][:3]
    except Exception as e:
        print("Warning: AI description generation failed:", e)
        return None

# takes the objects detected by yolo, distance and camera width to return feedback on object distance + give avoidance instructions
def get_accessibility_feedback(objects, ultrasonic_distance, frame_width):
    if enable_ai_description:
        ai_messages = generate_ai_description(objects, ultrasonic_distance, frame_width)
        if ai_messages:
            return ai_messages

    messages = []
    if ultrasonic_distance is not None:
        if ultrasonic_distance < 0.8:
            messages.append(f"Ultrasonic alert: obstacle very close ahead, {ultrasonic_distance:.1f} meters.")
        elif ultrasonic_distance < 1.5:
            messages.append(f"Ultrasonic alert: obstacle detected at {ultrasonic_distance:.1f} meters.")
        else:
            messages.append(f"Ultrasonic: no immediate obstacle detected. Distance {ultrasonic_distance:.1f} meters.")

    for obj in objects:
        label = obj["label"].lower()
        distance = obj["distance"] or "unknown distance"
        position = object_position(obj["box"], frame_width)
        if label in hazard_collision_labels:
            messages.append(f"Hazard: {label} detected {position} at {distance}. Keep a safe distance.")
        elif label in hazard_household_labels:
            messages.append(f"Obstacle: {label} detected {position} at {distance}. Navigate around it carefully.")
        else:
            messages.append(f"Detected: {label} {position} at {distance}.")

    if not messages:
        messages.append("No hazards detected in the current scene.")
    return messages

# plays sound (:
def play_sound(path):
    global sound_playing
    if sound_playing:
        return
    if not valid_sound_file(path):
        print(f"Warning: Cannot play sound, file missing: {path}")
        return

    def _play():
        global sound_playing
        sound_playing = True
        try:
            playsound(path)
        finally:
            sound_playing = False

    Thread(target=_play, daemon=True).start()

engine = init_tts() if pyttsx3 else None
ultrasonic_serial = init_ultrasonic_sensor()

# main loop
try:
    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("Error: Failed to read frame from camera")
            break

        annotated_frame = frame.copy()

        try:
            results = model.predict(frame, verbose=False)
            if results and len(results) > 0:
                r = results[0]
                annotated_frame = r.plot()
                last_detected_objects = process_detection_results(results, actual_height)
            else:
                last_detected_objects = []
        except Exception as e:
            print(f"Warning: model prediction failed: {e}")
            last_detected_objects = []

        # Read ultrasonic sensor and build accessibility messages
        ultrasonic_distance = read_ultrasonic_distance(ultrasonic_serial)
        accessibility_messages = get_accessibility_feedback(last_detected_objects, ultrasonic_distance, actual_width)

        for idx, message in enumerate(accessibility_messages[:3]):
            cv2.putText(annotated_frame, message, (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show frame
        cv2.imshow(window_name, annotated_frame)

        # Exit if window was closed via the X button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed, quitting...")
            break

        # Key handling with debouncing
        key = cv2.waitKey(10) & 0xFF
        if key != 255 and key != 0:
            now = time.time() * 1000
            if key != last_key or now - last_key_time > key_debounce_ms:
                print(f"Key pressed: {key}")
                last_key = key
                last_key_time = now

                if key == 32:
                    if accessibility_messages:
                        print("Accessibility feedback:")
                        for msg in accessibility_messages:
                            print(f"  - {msg}")
                        speak(" ".join(accessibility_messages[:3]))
                    else:
                        print("No hazards detected in the latest frame.")
                        speak("No hazards detected in the latest frame.")
                    play_sound(sound_path)
                elif key in (ord('q'), ord('Q'), 27):
                    print("Quitting...")
                    break
finally:
    capture.release()
    cv2.destroyAllWindows()
