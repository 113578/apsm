from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from endpoints import model_router
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
    """
    Возвращает статус работы API (сообщение об успешной работе).

    Returns
    -------
    StatusResponse
        Сообщение о статусе API (ключ 'message').
    """
    return StatusResponse(
        response={
            'message':
            'FastAPI работает! Используйте /docs для доступа к API.'
        }
    )


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
