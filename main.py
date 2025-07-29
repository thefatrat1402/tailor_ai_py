# === Install dependencies === 
# https://stackoverflow.com/questions/74970721/problem-with-installing-mediapipe-using-pip-on-windows
# python 3.10.11 for win
# pip install fastapi uvicorn opencv-python mediapipe numpy python-multipart

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import os

app = FastAPI()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def read_image(upload: UploadFile):
    image_data = upload.file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def measure(p1, p2, source, scale):
    return dist(source[p1], source[p2]) * scale

def interpolate_point(p1, p2, fraction):
    return (
        int(p1[0] + (p2[0] - p1[0]) * fraction),
        int(p1[1] + (p2[1] - p1[1]) * fraction)
    )

def body_slice_measure(front_points, side_points, scale, front_L, front_R, side_L1, side_L2, fraction, height_cm):
    pL = interpolate_point(front_points[front_L], front_points["LEFT_HIP"], fraction)
    pR = interpolate_point(front_points[front_R], front_points["RIGHT_HIP"], fraction)
    width_cm = dist(pL, pR) * scale

    s1 = interpolate_point(side_points[side_L1], side_points[side_L2], fraction)
    s2 = (s1[0] + 40, s1[1])
    depth_cm = dist(s1, s2) * scale

    circumference = width_cm * depth_cm
    return circumference, width_cm, depth_cm


# === Core API Endpoint ===
@app.post("/me/")
async def get_measurements(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    height_cm: float = Form(...)
):
    try:
        front_img = read_image(front_image)
        side_img = read_image(side_image)

        if front_img is None or side_img is None:
            return JSONResponse(content={"status": "error", "message": "Could not read one or both images."}, status_code=400)

        # === Detect Landmarks ===
        def get_keypoints(image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                return None
            h, w, _ = image.shape
            return {
                l.name: (int(result.pose_landmarks.landmark[l].x * w),
                         int(result.pose_landmarks.landmark[l].y * h))
                for l in mp_pose.PoseLandmark
            }

        front_points = get_keypoints(front_img)
        side_points = get_keypoints(side_img)

        if front_points is None or side_points is None:
            return JSONResponse(content={"status": "error", "message": "Pose not detected in one or both images."}, status_code=422)

        # === Scale Calculation ===
        front_top = front_points["NOSE"]
        front_bottom = (
            (front_points["LEFT_ANKLE"][0] + front_points["RIGHT_ANKLE"][0]) // 2,
            (front_points["LEFT_ANKLE"][1] + front_points["RIGHT_ANKLE"][1]) // 2,
        )
        pixel_height = dist(front_top, front_bottom)
        scale = height_cm / pixel_height

        # === Standard Measurements ===
        measurements = {
            "Height": height_cm,
            "Shoulder Width": measure("LEFT_SHOULDER", "RIGHT_SHOULDER", front_points, scale),
            "Hip Width": measure("LEFT_HIP", "RIGHT_HIP", front_points, scale),

            "Left Arm Length": measure("LEFT_SHOULDER", "LEFT_WRIST", front_points, scale),
            "Right Arm Length": measure("RIGHT_SHOULDER", "RIGHT_WRIST", front_points, scale),
            "Left Upper Arm": measure("LEFT_SHOULDER", "LEFT_ELBOW", front_points, scale),
            "Right Upper Arm": measure("RIGHT_SHOULDER", "RIGHT_ELBOW", front_points, scale),
            "Left Lower Arm": measure("LEFT_ELBOW", "LEFT_WRIST", front_points, scale),
            "Right Lower Arm": measure("RIGHT_ELBOW", "RIGHT_WRIST", front_points, scale),

            "Left Leg Length": measure("LEFT_HIP", "LEFT_ANKLE", front_points, scale),
            "Right Leg Length": measure("RIGHT_HIP", "RIGHT_ANKLE", front_points, scale),
            "Left Thigh": measure("LEFT_HIP", "LEFT_KNEE", front_points, scale),
            "Right Thigh": measure("RIGHT_HIP", "RIGHT_KNEE", front_points, scale),
            "Left Calf": measure("LEFT_KNEE", "LEFT_ANKLE", front_points, scale),
            "Right Calf": measure("RIGHT_KNEE", "RIGHT_ANKLE", front_points, scale),

            "Chest Depth": measure("LEFT_SHOULDER", "LEFT_HIP", side_points, scale) * 0.25,
            "Abdomen Depth": measure("LEFT_HIP", "LEFT_KNEE", side_points, scale) * 0.25,
            "Buttocks Depth": measure("RIGHT_HIP", "RIGHT_KNEE", side_points, scale) * 0.25,
            "Arm Depth": measure("LEFT_SHOULDER", "LEFT_ELBOW", side_points, scale) * 0.25,
        }

        # === Circumference Estimates ===
        chest_circ, chest_w, chest_d = body_slice_measure(front_points, side_points, scale,
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_HIP", 1/3, height_cm)

        ab_circ, ab_w, ab_d = body_slice_measure(front_points, side_points, scale,
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_HIP", 2/3, height_cm)

        waist_circ, waist_w, waist_d = body_slice_measure(front_points, side_points, scale,
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_HIP", 3/5, height_cm)

        measurements.update({
            "Chest Circumference": chest_circ,
            "Abdomen Circumference": ab_circ,
            "Waist Circumference": waist_circ
        })

        return {
            "status": "success",
            "height_cm": round(height_cm, 2),
            "measurements": {k: round(v, 2) for k, v in measurements.items()}
        }

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# === Run Locally ===
# To run: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
