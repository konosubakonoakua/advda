# Path hack
import sys
sys.path.insert(0, "../")

# Imports
from fastapi.middleware.cors import CORSMiddleware # CORS
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from typing import Optional
import uvicorn

from backend import cfg
from backend import router_yolov5
from backend import router_resnet

# Load the model

# Get the input shape for the model layer

# Define the FastAPI app
app = FastAPI()

# CORS
origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:23333",
    "http://localhost:4000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(router_yolov5.router)
app.include_router(router_resnet.router)

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
        return FileResponse(f"imgs/{filename}", media_type="image/"+filename.split(".")[-1])
    else:
        raise HTTPException(status_code=500, detail="wrong file name")


if __name__ == "__main__":
    # for p in sys.path:
    #     print(p)
    uvicorn.run(app=app, host=cfg.SERVER_HOST, port=cfg.SERVER_PORT)
