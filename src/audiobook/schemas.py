from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """
    Definition of a single audiobook chapter or section.
    """

    title: str = Field(..., description="The primary title of the chapter.")
    author: Optional[str] = Field(
        default=None,
        description="Specific author of this chapter (used for anthologies). Will be appended to the title.",
    )
    audio_path: Path = Field(
        ..., description="Absolute path to the lossless source audio file (.wav)."
    )


class AudiobookManifest(BaseModel):
    """
    The complete blueprint required to compile an audiobook.
    """

    title: str = Field(..., description="Global title of the audiobook.")
    author: str = Field(
        default="Various Authors", description="Global author, narrator, or compiler."
    )
    description: Optional[str] = Field(
        default=None,
        description="Synopsis of the audiobook to be injected as metadata.",
    )
    copyright: Optional[str] = Field(
        default=None, description="Copyright notice (e.g. '© 2026 Resona')."
    )
    cover_path: Optional[Path] = Field(
        default=None, description="Path to the cover image (.jpg or .png)."
    )
    chapters: List[Chapter] = Field(
        ..., min_length=1, description="Ordered list of chapters."
    )


class BuildResult(BaseModel):
    """
    The outcome of the compilation process.
    """

    success: bool = Field(
        ..., description="True if the .m4b was generated and tagged successfully."
    )
    output_path: Optional[Path] = Field(
        default=None, description="Path to the final .m4b file on disk if successful."
    )
    message: str = Field(
        ..., description="Status description or detailed error message if it failed."
    )
    processing_time_sec: Optional[float] = Field(
        default=None, description="Total execution time of the build process."
    )
