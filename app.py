from fastapi import FastAPI
from routers import pneumothorax_router

app = FastAPI()
app.include_router(pneumothorax_router.router, prefix='/pneumothorax')  # noqa


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Good to go'