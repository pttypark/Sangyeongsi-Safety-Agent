# HipHopWoo

A real-time safety management dashboard that monitors construction sites to ensure workers are wearing proper Personal Protective Equipment (PPE) and are not entering defined danger zones.
https://hub.ultralytics.com/models/UNwgMDZ5QChZ1tfzQS9S (yolo v11)

## Recent Improvements 

This branch introduces several significant improvements to the system:

- **Enhanced UI**: Modern interface with improved styling, icons, and better user feedback
- **Cross-Platform Support**: Complete compatibility with both Windows PCs and Jetson Nano devices
- **Video Source Selection**: New UI component to easily switch between camera and video files
- **Multiple Camera Support**: Dropdown menu to select from available cameras on the system
- **Jetson Nano Optimization**: Special camera handling with GStreamer for optimal performance on Jetson devices,Simplified Deployment: A single .py file streamlines distribution and setup, ideal for quick deployment across platforms.
Rapid Development and Testing: Centralized code enables faster iteration, debugging, and prototyping without managing multiple modules.
Embedded System Compatibility: Self-contained structure optimizes performance on resource-constrained devices like Jetson Nano.
- **Fixed Zone Management**: Improved zone creation and management with proper source-specific zones
- **Platform-Independent Paths**: Robust file handling across different operating systems
- **Command-Line Flexibility**: Added options to specify server IP and port for network accessibility


## Features

- **PPE Detection**: Automatically detects if workers are wearing helmets and safety vests
- **Danger Zone Monitoring**: Define custom danger zones and receive alerts when workers enter them
- **Multi-platform Support**: Works on both Windows and Linux systems, including NVIDIA Jetson Nano
- **Flexible Video Sources**: Use webcams, IP cameras, or pre-recorded videos
- **User-friendly Interface**: Modern web interface with real-time monitoring and alerts
- **Notification System**: Receive instant alerts for safety violations with screenshots
- **Statistics Dashboard**: Track safety metrics and compliance over time

## Requirements

- Python 3.7+ (Python 3.9 recommended)
- NVIDIA GPU recommended for faster inference (supports CPU mode as well)
- For Jetson Nano: JetPack 4.6+ with CUDA support

See `requirements.txt` for Python dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PPE-Detection-and-Danger-Zone-Monitoring-System.git
cd PPE-Detection-and-Danger-Zone-Monitoring-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model files and place them in the `models` directory:
   - `yolov11s.pt` (person detection model)
   - `ppe_v11s.pt` (PPE detection model)

## Usage

### Basic Usage
```bash
python main.py
```

This starts the system using your default camera.

### With a Video File
```bash
python main.py --Input "path/to/your/video.mp4"
```

### Specify Server IP and Port
```bash
python main.py --ip 0.0.0.0 --port 5000
```
Using 0.0.0.0 allows access from other devices on the network.

### Jetson Nano
On Jetson Nano, the system will automatically detect and use the onboard camera with optimized settings.

## UI Instructions

1. **Home Page**: Shows the live PPE detection feed
2. **Manage Page**: Define danger zones by drawing polygons on the video feed
3. **Statistics Page**: View PPE compliance statistics and reports
4. **Screenshots Page**: Access captured safety violation incidents

## How It Works

1. The system processes video frames in real-time using YOLOv11 models
2. First, people are detected in the frame
3. Then, PPE items (helmets and vests) are detected and associated with each person
4. The system checks if people are:
   - Wearing proper PPE
   - Located inside defined danger zones
5. Alerts are generated for any safety violations

## Cross-Platform Support

- **Windows**: Uses DirectShow for camera access
- **Linux/Jetson Nano**: Uses V4L2 and GStreamer for optimized camera access
- **Both**: Same web interface and detection capabilities

## Customization

- Adjust detection thresholds in the UI settings
- Draw custom danger zones specific to each video source
- Use multiple cameras or video sources

## Project Structure

- `/models` - YOLOv11 detection models
- `/static` - Web UI assets and captured screenshots
- `/templates` - HTML templates for the web interface
- `/video` - Sample videos for testing
- `/zone` - Saved danger zone configurations 
