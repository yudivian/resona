"""
Generates the text-based asset files required by FFmpeg for concatenation 
and chapter metadata injection.
"""

from pathlib import Path
from typing import List
from dataclasses import dataclass

from .schemas import AudiobookManifest
from .timing import ChapterTiming

@dataclass
class FFmpegAssets:
    """
    Holds the paths to the temporary text files generated for FFmpeg.
    """
    concat_file: Path
    metadata_file: Path


class MetadataWriter:
    """
    Translates Python data structures into FFmpeg's specific text formats.
    """

    @staticmethod
    def _escape_ffmpeg_string(value: str) -> str:
        """
        Escapes special characters in strings to comply with FFMETADATA format.
        FFmpeg requires escaping for: =, ;, #, \\ and newlines.
        """
        if not value:
            return ""
        
        escaped = str(value)
        for char in ['\\', '=', ';', '#', '\n']:
            escaped = escaped.replace(char, f'\\{char}')
        return escaped

    @classmethod
    def create_assets(cls, 
                      manifest: AudiobookManifest, 
                      timings: List[ChapterTiming], 
                      workspace_dir: Path) -> FFmpegAssets:
        """
        Writes the `concat.txt` and `FFMETADATA` files to the specified workspace.

        Args:
            manifest (AudiobookManifest): The global audiobook data.
            timings (List[ChapterTiming]): The calculated timeline.
            workspace_dir (Path): The directory where temporary files will be saved.

        Returns:
            FFmpegAssets: The paths to the generated files.
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        concat_path = workspace_dir / "concat.txt"
        ffmetadata_path = workspace_dir / "FFMETADATA.txt"

        with open(concat_path, 'w', encoding='utf-8') as f:
            for timing in timings:
                safe_path = str(timing.file_path.absolute()).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        with open(ffmetadata_path, 'w', encoding='utf-8') as f:
            f.write(";FFMETADATA1\n")
            
            f.write(f"title={cls._escape_ffmpeg_string(manifest.title)}\n")
            f.write(f"artist={cls._escape_ffmpeg_string(manifest.author)}\n")
            f.write(f"album={cls._escape_ffmpeg_string(manifest.title)}\n")
            
            if manifest.description:
                f.write(f"comment={cls._escape_ffmpeg_string(manifest.description)}\n")
            
            f.write("\n")

            for timing in timings:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={timing.start_ms}\n")
                f.write(f"END={timing.end_ms}\n")
                f.write(f"title={cls._escape_ffmpeg_string(timing.title)}\n")
                f.write("\n")

        return FFmpegAssets(concat_file=concat_path, metadata_file=ffmetadata_path)