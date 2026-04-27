from typing import Annotated

from pydantic import Field

BBox = Annotated[list[int], Field(min_length=4, max_length=4)]

