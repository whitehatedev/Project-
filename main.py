"""
AI-Based Urine Crystal Analyzer with Automatic Analysis Timer
Detects: CaOx Dihydrate, CaOx Monohydrate Ovoid, Phosphate
Version: 3.0 - Auto-stop after analysis period
"""

import sys
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from ultralytics import YOLO
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from threading import Lock
import os

# Import for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not installed. PDF generation disabled.")

class VideoThread(QThread):
    """
    Thread for handling video capture and processing
    """
    frame_ready = pyqtSignal(np.ndarray)
    detection_update = pyqtSignal(dict)
    fps_update = pyqtSignal(float)
    analysis_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = None
        self.model = None
        self.confidence_threshold = 0.5
        self.frame_skip = 2
        self.frame_count = 0
        self.lock = Lock()
        self.fps = 0
        self.fps_counter = 0
        self.fps_time = datetime.now()
        self.analysis_active = False
        self.analysis_start_time = None
        self.analysis_duration = 10  # seconds

    def set_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def set_confidence(self, conf):
        """Set confidence threshold"""
        self.confidence_threshold = conf

    def set_analysis_duration(self, seconds):
        """Set analysis duration in seconds"""
        self.analysis_duration = seconds

    def start_analysis(self):
        """Start the analysis period"""
        self.analysis_active = True
        self.analysis_start_time = datetime.now()

    def stop_analysis(self):
        """Stop the analysis period"""
        self.analysis_active = False
        self.analysis_start_time = None

    def run(self):
        """Main video processing loop"""
        self.cap = cv2.VideoCapture(1)

        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Calculate FPS
            self.fps_counter += 1
            if (datetime.now() - self.fps_time).seconds >= 1:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = datetime.now()
                self.fps_update.emit(self.fps)

            self.frame_count += 1

            # Check if analysis should auto-stop
            if self.analysis_active:
                elapsed = (datetime.now() - self.analysis_start_time).seconds
                if elapsed >= self.analysis_duration:
                    self.analysis_active = False
                    self.analysis_complete.emit()

            # Process every nth frame
            if self.frame_count % self.frame_skip == 0 and self.model:
                # Only process detections if analysis is active
                if self.analysis_active:
                    # Run inference
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)

                    # Process detections
                    counts = self.process_detections(frame, results[0])

                    # Emit detection update
                    self.detection_update.emit(counts)
                else:
                    # Just draw frame without processing detections
                    self.frame_ready.emit(frame)
                    continue

            # Emit frame for display
            self.frame_ready.emit(frame)

        # Cleanup
        if self.cap:
            self.cap.release()

    def process_detections(self, frame, result):
        """Process detections and draw bounding boxes"""
        counts = defaultdict(int)

        if result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Get class name
                class_name = self.model.names[cls]

                # Only count the 3 classes we care about
                if class_name in ['CaOx Dihydrate', 'CaOx Monohydrate Ovoid', 'Phosphate']:
                    counts[class_name] += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Add label with confidence
                    label = f"{class_name} ({conf:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1)-25), (int(x1)+label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return counts

    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()

class CrystalAnalyzer:
    """Medical analysis logic for crystal detection"""

    @staticmethod
    def analyze_caox_dihydrate(count):
        """Analyze CaOx Dihydrate crystals"""
        if count == 0:
            return "Normal", "No calcium oxalate dihydrate crystals detected", "Normal"
        elif count <= 3:
            return "Mild", "Mild presence of calcium oxalate dihydrate", "Mild"
        elif count <= 8:
            return "Moderate", "Moderate calcium oxalate dihydrate crystals - increased kidney stone risk", "Moderate ⚠️"
        else:
            return "High", "High calcium oxalate dihydrate - significant kidney stone risk", "High ⚠️"

    @staticmethod
    def analyze_caox_monohydrate(count):
        """Analyze CaOx Monohydrate Ovoid crystals"""
        if count == 0:
            return "Normal", "No calcium oxalate monohydrate crystals detected", "Normal"
        elif count <= 2:
            return "Mild", "Mild presence of calcium oxalate monohydrate", "Mild"
        elif count <= 5:
            return "Moderate", "Moderate calcium oxalate monohydrate - stone formation risk", "Moderate ⚠️"
        else:
            return "High", "High calcium oxalate monohydrate - severe stone formation risk", "High ⚠️"

    @staticmethod
    def analyze_phosphate(count):
        """Analyze Phosphate crystals"""
        if count == 0:
            return "Normal", "No phosphate crystals detected", "Normal"
        elif count <= 5:
            return "Mild", "Mild phosphate crystals - possible alkaline urine", "Mild"
        elif count <= 10:
            return "Moderate", "Moderate phosphate crystals - infection risk", "Moderate ⚠️"
        else:
            return "High", "High phosphate crystals - significant infection/alkaline urine", "High ⚠️"

    @staticmethod
    def get_recommendation(analyses):
        """Generate comprehensive recommendation"""
        recommendations = []
        severity_level = "Normal"

        for crystal_type, (status, detail, severity) in analyses.items():
            if "⚠️" in severity:
                severity_level = "Abnormal"
                recommendations.append(f"• {crystal_type}: {detail}")

        if not recommendations:
            recommendations.append("✓ All crystal levels within normal range.")
            recommendations.append("✓ Maintain adequate hydration (2-3 liters/day).")
            recommendations.append("✓ Continue regular health check-ups.")
        else:
            recommendations.insert(0, "Clinical Recommendations:")
            recommendations.append("• Increase water intake (minimum 2-3 liters/day)")
            recommendations.append("• Consider dietary modifications (reduce oxalate-rich foods)")
            recommendations.append("• Limit sodium and animal protein intake")
            recommendations.append("• Consult urologist for further evaluation")
            recommendations.append("• Repeat urinalysis in 2-4 weeks")

        return "\n".join(recommendations), severity_level

class UrineCrystalAnalyzer(QMainWindow):
    """Main GUI Window for Crystal Analysis"""

    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_counts = defaultdict(int)
        self.temp_counts = defaultdict(int)  # Temporary counts during analysis
        self.is_analyzing = False
        self.patient_name = ""
        self.patient_age = ""
        self.patient_gender = ""
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.check_analysis_progress)
        self.remaining_time = 0
        self.init_ui()

    def init_ui(self):
        """Initialize the UI with modern design"""
        self.setWindowTitle("AI Urine Crystal Analyzer - Automated Analysis System")
        self.setGeometry(100, 100, 1400, 800)

        # Set modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2f;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a4a6a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a7a;
            }
            QPushButton:pressed {
                background-color: #3a3a5a;
            }
            QPushButton:disabled {
                background-color: #2a2a3a;
                color: #6a6a8a;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #4a4a6a;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QComboBox {
                background-color: #2a2a3a;
                color: white;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
                padding: 5px;
            }
            QTextEdit {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #4a4a6a;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #4a4a6a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9e6e;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QProgressBar {
                border: 2px solid #4a4a6a;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #5a9e6e;
                border-radius: 3px;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Left panel - Video feed
        left_panel = QWidget()
        left_panel.setMinimumWidth(700)
        left_layout = QVBoxLayout(left_panel)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #4a4a6a; background-color: #000000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # Status bar
        status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #5a9e6e; font-weight: bold;")
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #ffaa6b; font-weight: bold;")
        self.confidence_label = QLabel("Confidence: 0.50")
        status_layout.addWidget(self.fps_label)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.confidence_label)
        left_layout.addLayout(status_layout)

        # Analysis progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Analysis Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        left_layout.addLayout(progress_layout)

        # Time display
        self.time_label = QLabel("Analysis Time: -- seconds")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #5a9e6e; font-size: 14px; font-weight: bold;")
        self.time_label.setVisible(False)
        left_layout.addWidget(self.time_label)

        # Confidence slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Detection Threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(25, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.conf_slider)
        left_layout.addLayout(conf_layout)

        # Duration selector
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Analysis Duration:"))
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["5 seconds", "10 seconds", "15 seconds", "20 seconds", "30 seconds"])
        self.duration_combo.setCurrentIndex(1)  # 10 seconds default
        duration_layout.addWidget(self.duration_combo)
        duration_layout.addStretch()
        left_layout.addLayout(duration_layout)

        main_layout.addWidget(left_panel)

        # Right panel - Controls and analysis
        right_panel = QWidget()
        right_panel.setMaximumWidth(450)
        right_layout = QVBoxLayout(right_panel)

        # Title
        title_label = QLabel("Automated Crystal Analysis")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #5a9e6e;")
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)

        # Patient information
        patient_group = QGroupBox("Patient Information")
        patient_layout = QGridLayout()

        patient_layout.addWidget(QLabel("Name:"), 0, 0)
        self.patient_name_edit = QLineEdit()
        self.patient_name_edit.setPlaceholderText("Enter patient name")
        patient_layout.addWidget(self.patient_name_edit, 0, 1)

        patient_layout.addWidget(QLabel("Age:"), 1, 0)
        self.patient_age_edit = QLineEdit()
        self.patient_age_edit.setPlaceholderText("Enter age")
        patient_layout.addWidget(self.patient_age_edit, 1, 1)

        patient_layout.addWidget(QLabel("Gender:"), 2, 0)
        self.patient_gender_combo = QComboBox()
        self.patient_gender_combo.addItems(["", "Male", "Female", "Other"])
        patient_layout.addWidget(self.patient_gender_combo, 2, 1)

        patient_group.setLayout(patient_layout)
        right_layout.addWidget(patient_group)

        # Crystal counts display
        counts_group = QGroupBox("Real-time Crystal Counts (during analysis)")
        counts_layout = QGridLayout()

        self.count_labels = {}
        crystals = ['CaOx Dihydrate', 'CaOx Monohydrate Ovoid', 'Phosphate']
        crystal_colors = ['#5a9e6e', '#4a9e8e', '#6a9e4e']

        for i, crystal in enumerate(crystals):
            label = QLabel(f"{crystal}:")
            label.setStyleSheet(f"font-weight: bold; color: {crystal_colors[i]};")
            value_label = QLabel("0")
            value_label.setStyleSheet(f"color: {crystal_colors[i]}; font-weight: bold; font-size: 18px;")
            counts_layout.addWidget(label, i, 0)
            counts_layout.addWidget(value_label, i, 1)
            self.count_labels[crystal] = value_label

        counts_group.setLayout(counts_layout)
        right_layout.addWidget(counts_group)

        # Control buttons
        buttons_group = QGroupBox("Controls")
        buttons_layout = QVBoxLayout()

        button_grid = QGridLayout()

        self.start_btn = QPushButton("▶ START ANALYSIS")
        self.start_btn.clicked.connect(self.start_analysis)
        self.start_btn.setStyleSheet("background-color: #5a9e6e; font-size: 14px; padding: 12px;")

        self.reset_btn = QPushButton("↺ RESET COUNTS")
        self.reset_btn.clicked.connect(self.reset_counts)

        self.pdf_btn = QPushButton("📄 GENERATE PDF REPORT")
        self.pdf_btn.clicked.connect(self.generate_pdf)
        self.pdf_btn.setEnabled(False)

        button_grid.addWidget(self.start_btn, 0, 0, 1, 2)
        button_grid.addWidget(self.reset_btn, 1, 0)
        button_grid.addWidget(self.pdf_btn, 1, 1)

        buttons_layout.addLayout(button_grid)
        buttons_group.setLayout(buttons_layout)
        right_layout.addWidget(buttons_group)

        # Analysis results
        analysis_group = QGroupBox("Analysis Results")
        analysis_layout = QVBoxLayout()
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(350)
        analysis_layout.addWidget(self.analysis_text)
        analysis_group.setLayout(analysis_layout)
        right_layout.addWidget(analysis_group)

        # Disclaimer
        disclaimer = QLabel(
            "⚠️ DISCLAIMER: AI-assisted analysis only.\n"
            "Must be verified by a certified medical professional.\n"
            "Camera remains active for continuous monitoring."
        )
        disclaimer.setStyleSheet("color: #ffaa6b; font-size: 10px; border: 1px solid #ffaa6b; padding: 5px;")
        disclaimer.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(disclaimer)

        main_layout.addWidget(right_panel)

        # Initialize video thread
        self.init_video_thread()

    def init_video_thread(self):
        """Initialize video processing thread"""
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_image)
        self.video_thread.detection_update.connect(self.update_counts)
        self.video_thread.fps_update.connect(self.update_fps)
        self.video_thread.analysis_complete.connect(self.on_analysis_complete)

        # Load model
        model_path = "best.pt"
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found",
                               f"Model file '{model_path}' not found!\n"
                               "Please ensure best.pt is in the same directory.")
        else:
            if not self.video_thread.set_model(model_path):
                QMessageBox.critical(self, "Error", "Failed to load model!")

        # Start video thread (camera always on)
        self.video_thread.start()

    def update_confidence(self):
        """Update confidence threshold"""
        value = self.conf_slider.value() / 100.0
        self.confidence_label.setText(f"Confidence: {value:.2f}")
        if self.video_thread:
            self.video_thread.set_confidence(value)

    def start_analysis(self):
        """Start the automated analysis"""
        if self.is_analyzing:
            return

        # Reset counts for new analysis
        self.reset_counts()

        # Get duration
        duration_text = self.duration_combo.currentText()
        duration = int(duration_text.split()[0])

        # Set analysis duration
        self.video_thread.set_analysis_duration(duration)
        self.remaining_time = duration

        # Start analysis
        self.is_analyzing = True
        self.video_thread.start_analysis()

        # Update UI
        self.start_btn.setEnabled(False)
        self.start_btn.setText("ANALYZING...")
        self.reset_btn.setEnabled(False)
        self.pdf_btn.setEnabled(False)
        self.status_label.setText(f"Status: Analyzing ({duration} sec)")
        self.status_label.setStyleSheet("color: #5a9e6e; font-weight: bold;")

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.time_label.setVisible(True)

        # Start timer for progress update
        self.analysis_timer.start(100)  # Update every 100ms

    def check_analysis_progress(self):
        """Update progress bar and time display"""
        if self.is_analyzing and self.video_thread.analysis_start_time:
            elapsed = (datetime.now() - self.video_thread.analysis_start_time).seconds
            remaining = max(0, self.video_thread.analysis_duration - elapsed)
            progress = int((elapsed / self.video_thread.analysis_duration) * 100)

            self.progress_bar.setValue(progress)
            self.time_label.setText(f"Remaining Time: {remaining} seconds")

            if remaining <= 0:
                self.analysis_timer.stop()

    def on_analysis_complete(self):
        """Handle analysis completion"""
        self.is_analyzing = False

        # Update UI
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ START ANALYSIS")
        self.reset_btn.setEnabled(True)
        self.pdf_btn.setEnabled(True)
        self.status_label.setText("Status: Analysis Complete")
        self.status_label.setStyleSheet("color: #5a9e6e; font-weight: bold;")

        # Hide progress elements
        self.progress_bar.setVisible(False)
        self.time_label.setVisible(False)

        # Perform analysis and display results
        self.perform_analysis()

        # Auto-scroll to results
        QTimer.singleShot(100, self.scroll_to_results)

    def scroll_to_results(self):
        """Scroll to show analysis results"""
        # This is a placeholder - results are already visible in the analysis panel
        pass

    def reset_counts(self):
        """Reset all counts"""
        self.current_counts = defaultdict(int)
        self.temp_counts = defaultdict(int)
        for crystal, label in self.count_labels.items():
            label.setText("0")
        self.analysis_text.clear()
        self.pdf_btn.setEnabled(False)
        self.status_label.setText("Status: Ready")
        self.status_label.setStyleSheet("color: #ffaa6b; font-weight: bold;")

    def update_image(self, cv_img):
        """Update video feed display"""
        if cv_img is not None:
            # Add status overlay if analyzing
            if self.is_analyzing:
                # Draw semi-transparent overlay
                overlay = cv_img.copy()
                cv2.rectangle(overlay, (0, 0), (cv_img.shape[1], 40), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, cv_img, 0.7, 0, cv_img)

                # Add text
                elapsed = 0
                if self.video_thread.analysis_start_time:
                    elapsed = (datetime.now() - self.video_thread.analysis_start_time).seconds
                    remaining = max(0, self.video_thread.analysis_duration - elapsed)
                    cv2.putText(cv_img, f"ANALYZING... {remaining}s remaining",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

    def update_counts(self, counts):
        """Update count display during analysis"""
        for crystal, count in counts.items():
            self.current_counts[crystal] += count
            if crystal in self.count_labels:
                self.count_labels[crystal].setText(str(self.current_counts[crystal]))

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps}")

    def perform_analysis(self):
        """Perform medical analysis on current counts"""
        analysis_results = {}

        # Analyze each crystal type
        analysis_results['CaOx Dihydrate'] = CrystalAnalyzer.analyze_caox_dihydrate(
            self.current_counts.get('CaOx Dihydrate', 0))
        analysis_results['CaOx Monohydrate Ovoid'] = CrystalAnalyzer.analyze_caox_monohydrate(
            self.current_counts.get('CaOx Monohydrate Ovoid', 0))
        analysis_results['Phosphate'] = CrystalAnalyzer.analyze_phosphate(
            self.current_counts.get('Phosphate', 0))

        # Generate recommendation
        recommendation, severity = CrystalAnalyzer.get_recommendation(analysis_results)

        # Get duration
        duration_text = self.duration_combo.currentText()

        # Build analysis text
        analysis_text = "=" * 60 + "\n"
        analysis_text += "URINE CRYSTAL ANALYSIS REPORT\n"
        analysis_text += "=" * 60 + "\n\n"

        # Patient info
        analysis_text += f"Patient: {self.patient_name_edit.text() or 'Not specified'}\n"
        analysis_text += f"Age: {self.patient_age_edit.text() or 'Not specified'}\n"
        analysis_text += f"Gender: {self.patient_gender_combo.currentText() or 'Not specified'}\n"
        analysis_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        analysis_text += f"Analysis Duration: {duration_text}\n"
        analysis_text += f"Overall Status: {severity}\n\n"

        analysis_text += "CRYSTAL DETECTION SUMMARY:\n"
        analysis_text += "-" * 60 + "\n"

        for crystal, (status, detail, severity_level) in analysis_results.items():
            count = self.current_counts.get(crystal, 0)
            analysis_text += f"\n{crystal}:\n"
            analysis_text += f"  Total Count: {count}\n"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  Interpretation: {detail}\n"

            # Add rate per minute
            duration_seconds = int(duration_text.split()[0])
            rate_per_min = (count / duration_seconds) * 60 if duration_seconds > 0 else 0
            analysis_text += f"  Rate: {rate_per_min:.1f} crystals/minute\n"

        analysis_text += "\n" + "-" * 60 + "\n"
        analysis_text += "RECOMMENDATIONS:\n"
        analysis_text += "-" * 60 + "\n"
        analysis_text += recommendation + "\n\n"

        analysis_text += "=" * 60 + "\n"
        analysis_text += "AI-Assisted Analysis - Must be verified by medical professional\n"

        self.analysis_text.setText(analysis_text)

    def generate_pdf(self):
        """Generate PDF report"""
        if not REPORTLAB_AVAILABLE:
            QMessageBox.warning(self, "PDF Generation Unavailable",
                               "ReportLab is not installed. Please install it using:\npip install reportlab")
            return

        try:
            filename = f"Crystal_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                        fontSize=18, alignment=TA_CENTER, spaceAfter=30,
                                        textColor=colors.HexColor('#5a9e6e'))
            heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                          fontSize=14, spaceAfter=10)
            normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                         fontSize=10, spaceAfter=5)

            # Title
            story.append(Paragraph("Urine Crystal Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Patient info
            duration_text = self.duration_combo.currentText()
            story.append(Paragraph(f"<b>Patient Name:</b> {self.patient_name_edit.text() or 'Not specified'}", normal_style))
            story.append(Paragraph(f"<b>Age:</b> {self.patient_age_edit.text() or 'Not specified'}", normal_style))
            story.append(Paragraph(f"<b>Gender:</b> {self.patient_gender_combo.currentText() or 'Not specified'}", normal_style))
            story.append(Paragraph(f"<b>Date & Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Paragraph(f"<b>Analysis Duration:</b> {duration_text}", normal_style))
            story.append(Spacer(1, 12))

            # Detection results table
            data = [['Crystal Type', 'Total Count', 'Rate/min', 'Status', 'Interpretation']]
            analysis_results = {}

            analysis_results['CaOx Dihydrate'] = CrystalAnalyzer.analyze_caox_dihydrate(
                self.current_counts.get('CaOx Dihydrate', 0))
            analysis_results['CaOx Monohydrate Ovoid'] = CrystalAnalyzer.analyze_caox_monohydrate(
                self.current_counts.get('CaOx Monohydrate Ovoid', 0))
            analysis_results['Phosphate'] = CrystalAnalyzer.analyze_phosphate(
                self.current_counts.get('Phosphate', 0))

            duration_seconds = int(duration_text.split()[0])

            for crystal, (status, detail, severity) in analysis_results.items():
                count = self.current_counts.get(crystal, 0)
                rate_per_min = (count / duration_seconds) * 60 if duration_seconds > 0 else 0
                data.append([crystal, str(count), f"{rate_per_min:.1f}", status, detail[:80] + '...' if len(detail) > 80 else detail])

            table = Table(data, colWidths=[120, 70, 70, 80, 180])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5a9e6e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 20))

            # Recommendations
            recommendation, severity = CrystalAnalyzer.get_recommendation(analysis_results)
            story.append(Paragraph("Clinical Recommendations", heading_style))
            story.append(Paragraph(recommendation.replace('\n', '<br/>'), normal_style))
            story.append(Spacer(1, 20))

            # Disclaimer
            story.append(Paragraph("DISCLAIMER", heading_style))
            story.append(Paragraph(
                "This is an AI-assisted analysis and must be verified by a certified medical professional. "
                "The results should not be used as the sole basis for diagnosis or treatment. "
                "Clinical correlation is essential for proper medical interpretation.",
                normal_style))

            # Build PDF
            doc.build(story)

            QMessageBox.information(self, "Success", f"Report saved as {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate PDF: {str(e)}")

    def closeEvent(self, event):
        """Handle application close"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = UrineCrystalAnalyzer()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()