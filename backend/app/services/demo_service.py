from backend.app.db.repositories import Repository
from backend.app.schemas.demo import DemoResetResponse
from backend.app.services.generated_asset_cleaner import GeneratedAssetCleaner


class DemoService:
    def __init__(
        self,
        *,
        repository: Repository,
        generated_asset_cleaner: GeneratedAssetCleaner | None = None,
    ) -> None:
        self._repository = repository
        self._generated_asset_cleaner = generated_asset_cleaner or GeneratedAssetCleaner()

    def reset(self) -> DemoResetResponse:
        removed = self._generated_asset_cleaner.clean()
        self._repository.reset()
        return DemoResetResponse(status="ok", generated_assets_removed=removed)
