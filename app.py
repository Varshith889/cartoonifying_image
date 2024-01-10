import cv2
import numpy as np

def cartoonify_image(img, cartoon_style, brightness=1.0, smoothness=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if cartoon_style == "Pencil Sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0, 0)
        blend = cv2.divide(gray, blur, scale=256)
        img_blend = cv2.cvtColor(blend, cv2.COLOR_GRAY2RGB)

    elif cartoon_style == "Cartoon-1":
        downsampled = cv2.pyrDown(img)
        blurred = cv2.bilateralFilter(downsampled, 9, 9, 7)
        upsampled = cv2.pyrUp(blurred)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        img_blend = cv2.bitwise_and(upsampled, upsampled, mask=edge)

    elif cartoon_style == "Cartoon-2":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 9)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        blurred = cv2.bilateralFilter(result, 4, 200, 200)
        img_blend = cv2.bitwise_and(blurred, blurred)

    # Adjust brightness and smoothness
    img_blend = cv2.convertScaleAbs(img_blend, alpha=brightness, beta=0)
    img_blend = cv2.bilateralFilter(img_blend, 9, smoothness, smoothness)

    return img_blend

if __name__ == "__main__":
    input_path = r'C:\python app1\a1.jpg'  # Use raw string to handle backslashes,input path specification
    output_path = 'output_cartoon_image.jpg'
    cartoon_style = "Pencil Sketch"
    cartoon_style = "Cartoon-1"
    brightness = 1.0
    smoothness = 5

    # Read the image
    input_image = cv2.imread(input_path)

    if input_image is not None:
        # Cartoonify the image
        cartoon_image = cartoonify_image(input_image, cartoon_style, brightness, smoothness)

        # Display and save the cartoonified image
        cv2.imshow('Cartoonify Image', cartoon_image)
        cv2.imwrite(output_path, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Failed to read the image from the path: {input_path}")
