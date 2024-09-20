# Vehicle Number Plate Detection System

This project implements a vehicle number plate detection system using a pre-trained PyTorch model for detection and Tesseract OCR for text recognition. The system captures video input, detects vehicle number plates, performs OCR, and logs the results (number plate and timestamp) in an Excel file.

## Features

- **Vehicle Number Plate Detection**: Detects number plates in real-time from a video feed.
- **OCR Recognition**: Extracts text from detected number plates using Tesseract OCR.
- **Logging**: Saves recognized plates along with timestamps to an Excel file.
- **Video Feed**: Displays the video feed with detected plates and recognized text in real-time.

## Requirements

To run this project, you need the following libraries and dependencies installed:

- Python 3.x
- OpenCV
- PyTorch
- pytesseract
- openpyxl
- Tesseract-OCR (Make sure to install Tesseract OCR on your machine)
  - Windows: [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`

## Installation

1. Clone this repository:
   
   git clone https://github.com/tusharahlawat/Vehicle-Number-Plate-Detection-System.git
 
   
2. Navigate to the project directory:

   cd vehicle-number-plate-detection

3. Install the required dependencies:
   pip install -r requirements.txt

4. Download and install Tesseract OCR:
   - Windows: [Tesseract Download](https://github.com/tesseract-ocr/tesseract/releases)
   - Linux: 
  
     sudo apt-get install tesseract-ocr

5. Ensure the PyTorch model is loaded from the correct path. Replace the path in the script with the actual path to your model:
   
   model_path = r"C:\path_to_your_model\license_plate_detector.pt"

## Usage

1. Run the detection script:

   python vehicle_detection.py

2. The system will open the video feed and start detecting vehicle number plates.

3. Press `q` to exit the program.

4. Recognized plates and their timestamps will be saved in the `vehicle_plates.xlsx` file.

## File Structure

- **vehicle_detection.py**: Main script for number plate detection and recognition.
- **requirements.txt**: Contains the list of Python dependencies.
- **license_plate_detector.pt**: Pre-trained PyTorch model for number plate detection.
- **vehicle_plates.xlsx**: Excel file where recognized plates and timestamps are saved.

## Notes

- The detection thresholds for the number plate size and aspect ratio can be adjusted in the `detect_number_plates` function based on your specific use case.
- Ensure that `pytesseract` is properly configured and the Tesseract executable is added to your system path.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

