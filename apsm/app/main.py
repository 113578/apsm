import uvicorn

from fastapi import FastAPI
from endpoints import *


app = FastAPI(
    title='APSM',
)
app.include_router(router=file_manager_router)
app.include_router(router=model_router)


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
