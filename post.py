import requests
import json

def get_body_measurements(front_image_path, side_image_path, height_cm, server_url="https://tailor-ai-py.onrender.com/me/"):
    try:
        files = {
            "front_image": open(front_image_path, "rb"),
            "side_image": open(side_image_path, "rb"),
        }
        data = {
            "height_cm": height_cm
        }

        response = requests.post(server_url, files=files, data=data)
        # Handle JSON response
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": response.text
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# === Example Usage ===
if __name__ == "__main__":
    front_path = "pics/gab_front.jpg"
    side_path = "pics/gab_side.jpg"
    height = 185.0

    result = get_body_measurements(front_path, side_path, height)

    # Print result as formatted JSON
    print(json.dumps(result, indent=2))
