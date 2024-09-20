import cv2
import pytesseract
import datetime
import os
import torch

# Load your trained PyTorch model (replace with your actual path)
model_path = r"C:\Users\HP\Desktop\DIP_Vehcile_detection_model\code.py\license_plate_detector.pt"
model = torch.load(model_path)
model.eval()  # Set model to evaluation mode

# Function to preprocess the image for number plate detection
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)  # Adaptive thresholding
    return thresh

# Function to detect number plates using contours and aspect ratio filtering
def detect_number_plates(image):
    thresh = preprocess_image(image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 2 < aspect_ratio < 5 and 30 < w < 150 and 10 < h < 70:  # Adjust thresholds as needed
            number_plates.append((x, y, w, h))
    return number_plates

# Function to perform OCR on a detected number plate region
def recognize_plate(image, plate_region):
    x, y, w, h = plate_region
    cropped_plate = image[y:y+h, x:x+w]
    # Improve text extraction using a combination of techniques for more robustness
    text = pytesseract.image_to_string(cropped_plate, config='--psm 6')  # Treat as single block of text
    text = cv2.text.ocr(cropped_plate, cv2.FONT_HERSHEY_COMPLEX, scale=1, thickness=2)[:, 0]  # Tesseract + OpenCV
    text = ''.join(c for c in text if c.isalnum())  # Keep only alphanumeric characters
    return text.strip()

# Function to save recognized plates and timestamps to an Excel file
def save_to_excel(plate_text, timestamp):
    excel_file = "vehicle_plates.xlsx"  # Adjust filename
    if not os.path.exists(excel_file):
        import openpyxl  # Import only when creating the file
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Plate Number", "Timestamp"])
        wb.save(excel_file)
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active
    ws.append([plate_text, timestamp])
    wb.save(excel_file)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to your video source (0 for webcam)

while True:
    ret, frame = cap.read()

    # Detect number plates
    number_plates = detect_number_plates(frame.copy())
    for plate in number_plates:
        x, y, w, h = plate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangle
        plate_text = recognize_plate(frame.copy(), plate)
        if plate_text:  # Check if text is extracted
            cv2.putText(frame, plate_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)  # Display plate text
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_to_excel(plate_text, timestamp)  # Save to Excel

    cv2.imshow("Vehicle Number Plates", frame)
            # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

print("Exiting Number Plate Detection Program...")

