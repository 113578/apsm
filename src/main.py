import os
import uvicorn

from fastapi import FastAPI
from api.file_manager import file_manager_router
from api.model import model_router

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(
    title='APSM',
)

app.include_router(router=file_manager_router)
app.include_router(router=model_router)


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
