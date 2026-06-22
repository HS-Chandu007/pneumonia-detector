import os
import uuid
import shutil

from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
from fastapi import File

from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from model import load_model, predict
from grad_cam import generate_gradcam


os.makedirs(
    "static/uploads",
    exist_ok=True
)

os.makedirs(
    "static/heatmaps",
    exist_ok=True
)


app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)



model = load_model()

classes = {
    0: "NORMAL",
    1: "PNEUMONIA"
}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post(
    "/predict",
    response_class=HTMLResponse
)
async def predict_xray(
    request: Request,
    file: UploadFile = File(...)
):

    try:

        ext = file.filename.split(".")[-1]

        unique_name = (
            f"{uuid.uuid4()}.{ext}"
        )

        upload_path = (
            f"static/uploads/{unique_name}"
        )

        with open(upload_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        pred, conf = predict(
            model,
            upload_path
        )

        heatmap_path = generate_gradcam(
            model,
            upload_path
        )

        return templates.TemplateResponse(
            request=request,
            name="result.html",
                context={
                    "prediction": classes[pred],
                    "confidence": round(conf * 100, 2),
                    "uploaded_image": upload_path,
                    "heatmap": heatmap_path
                }
)

    except Exception as e:

        return HTMLResponse(
            f"""
            <div class="error">
                Error: {str(e)}
            </div>
            """
        )