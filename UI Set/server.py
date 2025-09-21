import asyncio
import json
import random
import uvicorn
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def get():
    file_path = os.path.join(BASE_DIR, "index.html")
    with open(file_path) as f:
        return HTMLResponse(f.read())

# WebSocket endpoint for streaming data
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        time = 0
        while True:
            # 🚗 fake car positions
            cars = [
                {"id": i, "x": (i*10)+time, "y": (i*10)+time}
                for i in range(20)
            ]
            await ws.send_text(json.dumps(cars))
            await asyncio.sleep(0.01)  # simulate time step
            time += 1
            time %= 100
    except Exception as e:
        print("WebSocket closed:", e)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
