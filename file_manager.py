import logging.config
import os
from io import BytesIO

import aiofiles
from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logging.config.fileConfig("logging_conf")
logger = logging.getLogger()
app = FastAPI()

@app.post("/upload-file")
async def upload_csv(file: UploadFile = File(...))->dict[str, str]:
    try:
        filename = file.filename
        data = file.file.read().decode("utf-8")
        async with aiofiles.open(f"uploads/{filename}", "w") as f:
            await f.write(data)
        logger.info(f"Файл загружен: {filename}")
        return {"message": f"Файл {filename} успешно загружен"}

    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при загрузке файла")

@app.delete("/delete-file")
async def delete_file(file_name: str)->dict[str, str]:
    try:
        os.remove(f"uploads/{file_name}")
        logger.info(f"Файл удален: {file_name}")
        return {"message": f"Файл {file_name} успешно удален"}

    except Exception as e:
        logger.error(f"Ошибка при удалении файла {file_name}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при удалении файла")

@app.get("/get-file")
async def get_file(file_name: str)-> StreamingResponse:
    try:
        async with aiofiles.open(f"uploads/{file_name}", "rb") as f:
            data = await f.read()
        stream = BytesIO(data)
        response = StreamingResponse(iter([stream.getvalue()]),
                                     media_type="text/csv"
                                     )
        response.headers["Content-Disposition"] = f"attachment; filename={file_name}"
        logger.error(f"файл передан: {file_name}")
        return response

    except Exception as e:
        logger.error(f"Ошибка при получении файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при получении файла")


