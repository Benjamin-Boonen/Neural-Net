import asyncio
import json
import math
import uvicorn
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Canvas size (walls at edges)
WORLD_WIDTH = 600
WORLD_HEIGHT = 400

@app.get("/")
async def get():
    file_path = os.path.join(BASE_DIR, "index.html")
    with open(file_path) as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    car = {
        "id": 0,
        "x": 300.0,
        "y": 300.0,
        "angle": 0.0,      # car body heading
        "vel": 0.0,        # velocity
        "steer_angle": 0.0 # wheel angle
    }

    # Parameters
    accel_rate = 0.2
    brake_rate = 0.3
    friction = 0.02
    max_speed = 5.0
    max_steer = 0.6        # radians
    steer_speed = 0.05
    steer_return = 0.02    # auto-centering speed
    wheelbase = 30.0
    car_radius = 10
    bounce_factor = 0.3    # % of speed kept after bounce

    try:
        while True:
            throttle, steering = 0.0, 0.0

            # Receive inputs
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                command = json.loads(msg)
                throttle = float(command.get("throttle", 0.0))
                steering = float(command.get("steering", 0.0))
            except asyncio.TimeoutError:
                pass

            # Apply throttle
            if throttle > 0:
                car["vel"] += accel_rate * throttle
            elif throttle < 0:
                car["vel"] += brake_rate * throttle

            # Apply friction
            if car["vel"] > 0:
                car["vel"] -= friction
                if car["vel"] < 0: car["vel"] = 0
            elif car["vel"] < 0:
                car["vel"] += friction
                if car["vel"] > 0: car["vel"] = 0

            # Clamp velocity
            car["vel"] = max(-max_speed, min(max_speed, car["vel"]))

            # Update steering
            if steering != 0:
                car["steer_angle"] += steering * steer_speed
            else:
                # Auto-center steering toward 0
                if car["steer_angle"] > 0:
                    car["steer_angle"] = max(0, car["steer_angle"] - steer_return)
                elif car["steer_angle"] < 0:
                    car["steer_angle"] = min(0, car["steer_angle"] + steer_return)

            car["steer_angle"] = max(-max_steer, min(max_steer, car["steer_angle"]))

            # Update heading if moving
            if abs(car["vel"]) > 0:
                car["angle"] += (car["vel"] / wheelbase) * math.tan(car["steer_angle"])

            # Move car
            dx = math.cos(car["angle"])
            dy = math.sin(car["angle"])
            car["x"] += dx * car["vel"]
            car["y"] += dy * car["vel"]

            # -------------------
            # WALL COLLISION (bounce back)
            # -------------------
            if car["x"] < car_radius:
                car["x"] = car_radius
                car["vel"] = -car["vel"] * bounce_factor
            elif car["x"] > WORLD_WIDTH - car_radius:
                car["x"] = WORLD_WIDTH - car_radius
                car["vel"] = -car["vel"] * bounce_factor

            if car["y"] < car_radius:
                car["y"] = car_radius
                car["vel"] = -car["vel"] * bounce_factor
            elif car["y"] > WORLD_HEIGHT - car_radius:
                car["y"] = WORLD_HEIGHT - car_radius
                car["vel"] = -car["vel"] * bounce_factor

            # Send state
            car_state = {
                "id": car["id"],
                "x": car["x"],
                "y": car["y"],
                "dir": [dx, dy],
                "steer_angle": car["steer_angle"]
            }
            await ws.send_text(json.dumps([car_state]))
            await asyncio.sleep(0.01)

    except Exception as e:
        print("WebSocket closed:", e)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
