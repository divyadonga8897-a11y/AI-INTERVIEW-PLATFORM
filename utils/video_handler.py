import cv2
import numpy as np
import time
import os

# Try to import mediapipe for advanced mesh tracking
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    pass

class VideoAnalyzer:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.prev_face_pos = None
        self.last_time = time.time()
        self.jitter_scores = []
        self.mp_face_mesh = None
        self.face_mesh = None
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("VideoAnalyzer: MediaPipe FaceMesh initialized successfully.")
            except Exception as e:
                print(f"VideoAnalyzer: Failed to initialize MediaPipe: {e}. Falling back to Haar Cascades.")
                self.init_haar_cascades()
        else:
            self.init_haar_cascades()

    def init_haar_cascades(self):
        """Initializes OpenCV Haar Cascade Classifiers."""
        # Find paths to default classifiers
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        if os.path.exists(face_path):
            self.face_cascade = cv2.CascadeClassifier(face_path)
        if os.path.exists(eye_path):
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
            
        print("VideoAnalyzer: Haar Cascades initialized.")

    def analyze_frame(self, frame):
        """
        Analyzes a single frame from the webcam.
        Draws HUD overlay, returns processed frame and metrics dict:
        {
            'eye_contact': 0-100,
            'expression': 'Confident' | 'Nervous' | 'Neutral',
            'confidence_score': 0-100,
            'face_detected': bool
        }
        """
        if frame is None:
            return None, {'eye_contact': 0, 'expression': 'No Feed', 'confidence_score': 0, 'face_detected': False}
            
        h, w, _ = frame.shape
        metrics = {
            'eye_contact': 80,
            'expression': 'Neutral',
            'confidence_score': 80,
            'face_detected': False
        }
        
        # Make a copy of the frame to draw on
        overlay = frame.copy()
        
        # 1. MediaPipe Mode
        if HAS_MEDIAPIPE and self.face_mesh:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    metrics['face_detected'] = True
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Track a key point (nose tip, index 4) for jitter analysis
                    nose = landmarks[4]
                    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                    
                    # Calculate movement speed (jitter)
                    current_time = time.time()
                    dt = current_time - self.last_time
                    self.last_time = current_time
                    
                    jitter = 0
                    if self.prev_face_pos is not None and dt > 0:
                        dx = nose_x - self.prev_face_pos[0]
                        dy = nose_y - self.prev_face_pos[1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        jitter = dist / dt # pixels per second
                        
                    self.prev_face_pos = (nose_x, nose_y)
                    
                    # Smooth jitter score
                    self.jitter_scores.append(jitter)
                    if len(self.jitter_scores) > 30:
                        self.jitter_scores.pop(0)
                    avg_jitter = np.mean(self.jitter_scores)
                    
                    # Determine expression and confidence based on physical jitter
                    # If candidate is moving around too quickly, they are nervous. If steady, they are confident.
                    if avg_jitter > 250:
                        metrics['expression'] = 'Nervous'
                        metrics['confidence_score'] = max(30, int(100 - avg_jitter/5))
                    elif avg_jitter < 80:
                        metrics['expression'] = 'Confident'
                        metrics['confidence_score'] = min(98, int(85 + (80 - avg_jitter)/4))
                    else:
                        metrics['expression'] = 'Neutral'
                        metrics['confidence_score'] = int(80 - (avg_jitter - 80)/10)
                        
                    # Eye-tracking (L: 33, R: 263 indices range or simple coordinate check)
                    # For simplify, eye contact is scored on centering. If user looks away, nose position shifts.
                    # Normal range for centering
                    center_offset_x = abs(nose.x - 0.5)
                    center_offset_y = abs(nose.y - 0.5)
                    
                    if center_offset_x > 0.15 or center_offset_y > 0.2:
                        # Looking away
                        metrics['eye_contact'] = max(10, int(90 - (center_offset_x - 0.15) * 300))
                    else:
                        metrics['eye_contact'] = min(100, int(95 - center_offset_x * 50))
                        
                    # Draw HUD overlays
                    # Draw face box boundary
                    xs = [int(l.x * w) for l in landmarks]
                    ys = [int(l.y * h) for l in landmarks]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    
                    # Draw futuristic corners instead of full rectangle
                    length = 20
                    color = (0, 255, 0) if metrics['expression'] == 'Confident' else ((0, 165, 255) if metrics['expression'] == 'Neutral' else (0, 0, 255))
                    
                    # Top Left
                    cv2.line(overlay, (xmin, ymin), (xmin + length, ymin), color, 2)
                    cv2.line(overlay, (xmin, ymin), (xmin, ymin + length), color, 2)
                    # Top Right
                    cv2.line(overlay, (xmax, ymin), (xmax - length, ymin), color, 2)
                    cv2.line(overlay, (xmax, ymin), (xmax, ymin + length), color, 2)
                    # Bottom Left
                    cv2.line(overlay, (xmin, ymax), (xmin + length, ymax), color, 2)
                    cv2.line(overlay, (xmin, ymax), (xmin, ymax - length), color, 2)
                    # Bottom Right
                    cv2.line(overlay, (xmax, ymax), (xmax - length, ymax), color, 2)
                    cv2.line(overlay, (xmax, ymax), (xmax, ymax - length), color, 2)
                    
                    # Draw face mesh points (subset to avoid clutter)
                    for idx in [33, 133, 362, 263, 1, 61, 291]:  # eyes, nose, mouth corners
                        pt = landmarks[idx]
                        cv2.circle(overlay, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 255), -1)
                        
                    # Add Text HUD
                    cv2.putText(overlay, f"STATE: {metrics['expression']}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
            except Exception as e:
                print(f"Error in MediaPipe frame process: {e}")
                
        # 2. Haar Cascade Fallback Mode (if no MediaPipe or fails)
        if not metrics['face_detected'] and self.face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                metrics['face_detected'] = True
                (x, y, fw, fh) = faces[0]
                
                # Face center tracking
                face_center_x = x + fw // 2
                face_center_y = y + fh // 2
                
                # Jitter calculation
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                
                jitter = 0
                if self.prev_face_pos is not None and dt > 0:
                    dx = face_center_x - self.prev_face_pos[0]
                    dy = face_center_y - self.prev_face_pos[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    jitter = dist / dt
                    
                self.prev_face_pos = (face_center_x, face_center_y)
                
                self.jitter_scores.append(jitter)
                if len(self.jitter_scores) > 30:
                    self.jitter_scores.pop(0)
                avg_jitter = np.mean(self.jitter_scores)
                
                # Setup expression & confidence
                if avg_jitter > 150:
                    metrics['expression'] = 'Nervous'
                    metrics['confidence_score'] = max(40, int(100 - avg_jitter/3))
                elif avg_jitter < 50:
                    metrics['expression'] = 'Confident'
                    metrics['confidence_score'] = min(98, int(85 + (50 - avg_jitter)/2))
                else:
                    metrics['expression'] = 'Neutral'
                    metrics['confidence_score'] = int(80 - (avg_jitter - 50)/5)
                
                # Eye-tracking fallback: scan for eyes inside the face region
                roi_gray = gray[y:y+fh, x:x+fw]
                roi_color = overlay[y:y+fh, x:x+fw]
                
                eyes = []
                if self.eye_cascade:
                    # Restrict eye scanning to upper half of face
                    eye_roi_gray = roi_gray[0:int(fh*0.65), :]
                    eyes = self.eye_cascade.detectMultiScale(eye_roi_gray, 1.15, 3, minSize=(15, 15))
                
                # Eye contact scoring:
                # If 2 eyes detected, high score. 1 eye, medium. 0 eyes, low (candidate looking away)
                if len(eyes) >= 2:
                    metrics['eye_contact'] = 90
                elif len(eyes) == 1:
                    metrics['eye_contact'] = 60
                else:
                    metrics['eye_contact'] = 30
                    
                # Draw boxes
                # Draw face corner HUD
                color = (0, 255, 0) if metrics['expression'] == 'Confident' else ((0, 165, 255) if metrics['expression'] == 'Neutral' else (0, 0, 255))
                length = 15
                cv2.line(overlay, (x, y), (x + length, y), color, 2)
                cv2.line(overlay, (x, y), (x, y + length), color, 2)
                cv2.line(overlay, (x + fw, y), (x + fw - length, y), color, 2)
                cv2.line(overlay, (x + fw, y), (x + fw, y + length), color, 2)
                cv2.line(overlay, (x, y + fh), (x + length, y + fh), color, 2)
                cv2.line(overlay, (x, y + fh), (x, y + fh - length), color, 2)
                cv2.line(overlay, (x + fw, y + fh), (x + fw - length, y + fh), color, 2)
                cv2.line(overlay, (x + fw, y + fh), (x + fw, y + fh - length), color, 2)
                
                # Draw eye boxes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                    
                cv2.putText(overlay, f"HUD: {metrics['expression']}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 3. Add General HUD Layout
        # Draw scanning grid line moving up and down
        scan_y = int((time.time() * 80) % h)
        cv2.line(overlay, (0, scan_y), (w, scan_y), (0, 255, 0), 1)
        
        # Add visual overlay stats in corners
        hud_box_w = 150
        hud_box_h = 75
        # Top-right dashboard
        cv2.rectangle(overlay, (w - hud_box_w - 10, 10), (w - 10, hud_box_h + 10), (15, 23, 42), -1)
        cv2.rectangle(overlay, (w - hud_box_w - 10, 10), (w - 10, hud_box_h + 10), (51, 65, 85), 1)
        
        cv2.putText(overlay, f"EYES: {metrics['eye_contact']}%", (w - hud_box_w, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(overlay, f"CONF: {metrics['confidence_score']}%", (w - hud_box_w, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(overlay, f"FACE: {'OK' if metrics['face_detected'] else 'MISSING'}", (w - hud_box_w, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255) if metrics['face_detected'] else (0, 0, 255), 1)
        
        return overlay, metrics
