import uvicorn

from fastapi import FastAPI
from http import HTTPStatus
from endpoints import *
from schemas import StatusResponse


app = FastAPI(
    title='APSM',
)
app.include_router(router=model_router)


@app.get(
    '/',
    response_model=StatusResponse,
    status_code=HTTPStatus.OK
)
async def root() -> StatusResponse:
    return StatusResponse(
        response={
            'message':
            'FastAPI работает! Используйте /docs для доступа к API.'
        }
    )


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
