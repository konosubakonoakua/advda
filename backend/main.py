import sys
sys.path.insert(0, "../")
import uvicorn
from typing import Optional
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # CORS
from backend import router_resnet
from backend import router_yolov5
from backend import cfg
from backend.log import setup_logging, LOG_LEVEL, Server, Config
from backend.applog import read_logging_config, setup_logging

logconfig_dict = read_logging_config("applog/logging.yaml")
setup_logging(logconfig_dict)


# Define the FastAPI app
app = FastAPI()

# move to cfg.py
# # CORS
# origins = [
#     # "http://localhost.tiangolo.com",
#     # "https://localhost.tiangolo.com",
#     "http://localhost:40000",
#     "http://localhost:23333",
#     "http://192.168.224.1",
#     "http://127.0.0.1",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(router_yolov5.router)
app.include_router(router_resnet.router)

# Mount folder
# app.mount("/static", StaticFiles(directory="../web/static"), name="static")


# Define the main route
@app.get('/')
async def root_route():
    # return "Welcome to this demo."
    return RedirectResponse("/yolo/upload")


# images response
@app.get("/imgs/{filename}")
async def get_img_file(filename: Optional[str] = None):
    if filename is not None:
        # img: PIL.Image.Image = Image.open(f"web/imgs/{filename}")
        return FileResponse(f"imgs/{filename}", media_type="image/"+filename.split(".")[-1])
    else:
        raise HTTPException(status_code=500, detail="wrong file name")


if __name__ == "__main__":
    # setup_logging()
    uvicorn.run(app=app, host=cfg.SERVER_HOST, port=cfg.SERVER_PORT)

    # server = Server(
    #     Config(
    #         "main:app",
    #         host=cfg.SERVER_HOST,
    #         port=cfg.SERVER_PORT,
    #         log_level=LOG_LEVEL,
    #     ),
    # )

    # # setup logging last, to make sure no library overwrites it
    # # (they shouldn't, but it happens)
    # setup_logging()

    # server.run()
