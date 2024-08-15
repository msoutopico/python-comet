from code.routers.mtpe import router as scores_router

from fastapi import FastAPI

app = FastAPI()

app.include_router(scores_router, prefix="/api")
