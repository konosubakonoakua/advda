# Imports
from fastapi import FastAPI, HTTPException
import uvicorn
from starlette.responses import FileResponse

from backend import router_yolov5
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

# Load the model

# Get the input shape for the model layer

# Define the FastAPI app
app = FastAPI()

# Add routers
app.include_router(router_yolov5.router)

# Mount folder
app.mount("/static", StaticFiles(directory="../web/static"), name="static")


# Define the main route
@app.get('/')
async def root_route():
    # return "Welcome to this demo."
    return RedirectResponse("/yolo/upload")


# images response
@app.get("/imgs/pred/{filename}")
async def get_img_file(filename: Optional[str] = None):
    if filename is not None:
        # img: PIL.Image.Image = Image.open(f"web/imgs/{filename}")
        return FileResponse(f"imgs/{filename}", media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail="wrong file name")


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8888)
