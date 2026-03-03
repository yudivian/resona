"""
Core FFmpeg integration for the audiobook pipeline.
Responsible for concatenating audio streams and multiplexing metadata into a single container.
"""

import subprocess
import logging
from pathlib import Path

from .metadata import FFmpegAssets

logger = logging.getLogger("AudiobookAssembler")
logger.setLevel(logging.INFO)

class AudiobookAssembler:
    """
    Executes the FFmpeg assembly graph to build the M4B/M4A container.
    """

    @staticmethod
    def assemble(assets: FFmpegAssets, output_path: Path) -> Path:
        """
        Runs the FFmpeg concat demuxer and maps the FFMETADATA file into the output.

        Args:
            assets (FFmpegAssets): The paths to the concat and metadata text files.
            output_path (Path): Where the final assembled audio file should be saved.

        Returns:
            Path: The path to the successfully generated audio file.
            
        Raises:
            RuntimeError: If FFmpeg encounters an error during execution.
        """
        logger.info(f"Starting FFmpeg assembly into {output_path.name}")
        
        cmd = [
            "ffmpeg",
            "-y",                               
            "-f", "concat",
            "-safe", "0",
            "-i", str(assets.concat_file),      
            "-i", str(assets.metadata_file),    
            "-map", "0:a",                      
            "-map_metadata", "1",               
            "-c:a", "aac",                      
            "-b:a", "128k",                     
            "-loglevel", "error",               
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
                
            if not output_path.exists():
                raise RuntimeError("FFmpeg completed but output file is missing.")
                
            logger.info("FFmpeg assembly completed successfully.")
            return output_path
            
        except Exception as e:
            logger.error(f"FFmpeg crashed: {str(e)}")
            raise RuntimeError(f"Failed to assemble audiobook: {str(e)}")