import asyncio
import websockets
import json
import base64

async def test_ws():
    uri = "ws://localhost:8000/ws/video"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            # Receive 3 frames
            for i in range(3):
                msg = await websocket.recv()
                data = json.loads(msg)
                if "error" in data:
                    print(f"Server error: {data['error']}")
                    break
                
                print(f"Received {data.get('type')} msg, frame size: {len(data.get('frame', ''))}")
                
            print("Done.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
