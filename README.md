#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```python

image = cv2.imread("road.jpg")

```

---

###  Step 3: Convert to Grayscale

```python

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

```

---

###  Step 4: Display Images

```
# 1. Original Image
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("1. Original Image")
plt.axis("off")
# 2. Grayscale Image
plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("2. Grayscale Image")
plt.axis("off"
```

---

###  Step 5: Thresholding

```python
threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
```

---

###  Step 6: Region of Interest (ROI)

```
height, width = image.shape[:2]

mask = np.zeros_like(gray)

polygon = np.array([[
    (0, height),
    (width, height),
    (int(width * 0.55), int(height * 0.55)),
    (int(width * 0.45), int(height * 0.55))
]], np.int32)

cv2.fillPoly(mask, polygon, 255)

roi_image = cv2.bitwise_and(image, image, mask=mask)
```

---

### Step 7: Edge Detection (Canny)

```

edges = cv2.Canny(gray, 50, 150)

```

---

###  Step 8: Gaussian Blur

```
# Apply Gaussian Blur

blurred = cv2.GaussianBlur(edges, (5, 5), 0)
```

---

###  Step 9: Hough Transform

```
# Detect yellow and white lane markings using HSV color

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Yellow line mask
lower_yellow = np.array([15, 60, 100])
upper_yellow = np.array([40, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# White line mask
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 90, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Road ROI
height, width = image.shape[:2]

road_mask = np.zeros((height, width), dtype=np.uint8)

road_polygon = np.array([[
    (0, height),
    (width, height),
    (500, 145),
    (280, 145)
]], np.int32)

cv2.fillPoly(road_mask, road_polygon, 255)

yellow_mask = cv2.bitwise_and(yellow_mask, road_mask)
white_mask = cv2.bitwise_and(white_mask, road_mask)
```

---

### Step 10: Lane Detection Logic

```
# Create output image
line_image = image.copy()

# Function to fit and draw a smooth curved line
def draw_curve(mask, x_min, x_max, color, thickness=5):

    ys, xs = np.where(mask > 0)

    # Select only pixels in the required road-line region
    condition = (xs >= x_min) & (xs <= x_max) & (ys >= 145)

    xs = xs[condition]
    ys = ys[condition]

    if len(xs) > 50:

        # Fit curved line: x = a*y^2 + b*y + c
        coefficients = np.polyfit(ys, xs, 2)

        y_values = np.linspace(150, height - 1, 200)
        x_values = np.polyval(coefficients, y_values)

        points = np.column_stack(
            (x_values, y_values)
        ).astype(np.int32)

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)

        cv2.polylines(
            line_image,
            [points.reshape(-1, 1, 2)],
            False,
            color,
            thickness,
            cv2.LINE_AA
        )


# Yellow center road line
draw_curve(
    yellow_mask,
    300,
    450,
    (0, 255, 255),
    6
)

# Left white road edge
draw_curve(
    white_mask,
    50,
    250,
    (255, 255, 255),
    6
)




# Final result
final_image = line_image.copy()

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
```
## Final program
~~~
plt.figure(figsize=(16, 12))

# 1. Original
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 2. Grayscale
plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# 3. Threshold
plt.subplot(3, 3, 3)
plt.imshow(threshold, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")

# 4. ROI
plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
plt.title("ROI Masked Image")
plt.axis("off")

# 5. Edges
plt.subplot(3, 3, 5)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")

# 6. Smoothed
plt.subplot(3, 3, 6)
plt.imshow(blurred, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")


# 7. Detected Lines
plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Road Lines")
plt.axis("off")

# 8. Final Lane Detection
plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.tight_layout()
plt.show()
~~~
---

##  Output

<img width="1366" height="919" alt="image" src="https://github.com/user-attachments/assets/d28be407-89d3-4ee3-ac9f-fe87af87a3ef" />


---

##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name:** POTHU SUMANTH
* **Register No:** 212224240115
