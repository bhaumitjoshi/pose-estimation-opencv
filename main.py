import cv2
import numpy as np


def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length=50.0):
    """
    Draw 3D axes on the image.
    """
    axis_points_3d = np.float32([
        [0, 0, 0],              # origin
        [axis_length, 0, 0],    # X
        [0, axis_length, 0],    # Y
        [0, 0, -axis_length]    # Z (negative so it shows forward in many image conventions)
    ])

    img_points, _ = cv2.projectPoints(
        axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    img_points = img_points.reshape(-1, 2).astype(int)

    origin = tuple(img_points[0])
    x_axis = tuple(img_points[1])
    y_axis = tuple(img_points[2])
    z_axis = tuple(img_points[3])

    cv2.line(img, origin, x_axis, (0, 0, 255), 3)   # X - red
    cv2.line(img, origin, y_axis, (0, 255, 0), 3)   # Y - green
    cv2.line(img, origin, z_axis, (255, 0, 0), 3)   # Z - blue

    cv2.putText(img, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img


def main():
    # Example 3D object points (square marker corners in mm)
    object_points = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 100.0, 0.0],
        [0.0, 100.0, 0.0]
    ], dtype=np.float32)

    # Example corresponding 2D image points (pixels)
    image_points = np.array([
        [320.0, 240.0],
        [420.0, 250.0],
        [410.0, 350.0],
        [300.0, 340.0]
    ], dtype=np.float32)

    # Example camera intrinsics
    fx = 800.0
    fy = 800.0
    cx = 320.0
    cy = 240.0

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # Create blank image for visualization
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Draw input image points
    for pt in image_points.astype(int):
        cv2.circle(img, tuple(pt), 5, (0, 0, 0), -1)

    img = draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length=80.0)

    cv2.imshow("Pose Estimation Demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()