from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.routes import demo, jobs, scenes, stories


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount(
        settings.static_url_prefix,
        StaticFiles(directory=settings.static_dir),
        name="static",
    )

    app.include_router(stories.router)
    app.include_router(scenes.router)
    app.include_router(jobs.router)
    app.include_router(demo.router)
    return app


app = create_app()
