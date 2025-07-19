import qrcode
from PIL import Image

QR_CODE_DATA = "MRI_HEAD_MOTION_TRACKER_V1.0"
OUTPUT_FILENAME = "mri_tracker_qr.png"

# --- Physical Size Configuration ---
TARGET_WIDTH_INCHES = 1.5
DPI = 300

def generate_fixed_size_qr_code(data, filename, inches, dpi):
    target_pixel_width = int(inches * dpi)

    qr_check = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, border=0)
    qr_check.add_data(data)
    qr_check.make(fit=True)
    total_modules = qr_check.modules_count
    
    box_size = int(target_pixel_width / total_modules)
    if box_size == 0:
        box_size = 1

    qr = qrcode.QRCode(
        version=qr_check.version,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size, # Use the calculated size here
        border=0, 
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white") # No box_size here
    resized_img = img.resize((target_pixel_width, target_pixel_width), Image.NEAREST)
    resized_img.save(filename, dpi=(dpi, dpi))

    print(f"QR Code saved as '{filename}'")
    print(f"The image is set to print at a width of {inches} inches ({resized_img.size[0]} x {resized_img.size[1]}px)")

if __name__ == "__main__":
    generate_fixed_size_qr_code(QR_CODE_DATA, OUTPUT_FILENAME, TARGET_WIDTH_INCHES, DPI)
