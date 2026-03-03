"""
Post-processing metadata injection using Mutagen.
Handles Apple-specific Audiobook tags (stik) and binary cover art insertion.
"""

import logging
from pathlib import Path
from mutagen.mp4 import MP4, MP4Cover

from .schemas import AudiobookManifest

logger = logging.getLogger("AudiobookTagger")
logger.setLevel(logging.INFO)

class AudiobookTagger:
    """
    Injects ID3/MPEG-4 atoms into the compiled M4B file.
    """

    @staticmethod
    def apply_tags(m4b_path: Path, manifest: AudiobookManifest) -> None:
        """
        Opens the M4B file and writes the rich metadata and cover art.

        Args:
            m4b_path (Path): The path to the assembled M4B file.
            manifest (AudiobookManifest): The blueprint containing the metadata to inject.

        Raises:
            Exception: If Mutagen fails to read or write the tags.
        """
        logger.info(f"Applying rich metadata to {m4b_path.name}")
        
        try:
            audio = MP4(str(m4b_path))
            
            audio.tags['stik'] = [2]
            
            audio.tags['\xa9nam'] = [manifest.title]                  
            audio.tags['\xa9alb'] = [manifest.title]                  
            audio.tags['\xa9ART'] = [manifest.author]                 
            audio.tags['aART'] = [manifest.author]                    
            
            if manifest.description:
                audio.tags['desc'] = [manifest.description]           
            
            if manifest.copyright:
                audio.tags['cprt'] = [manifest.copyright]             

            if manifest.cover_path and manifest.cover_path.exists():
                with open(manifest.cover_path, "rb") as img_file:
                    img_data = img_file.read()
                    
                    suffix = manifest.cover_path.suffix.lower()
                    img_format = MP4Cover.FORMAT_PNG if suffix == ".png" else MP4Cover.FORMAT_JPEG
                    
                    audio.tags['covr'] = [MP4Cover(img_data, imageformat=img_format)]
                    logger.info("Cover art injected successfully.")

            audio.save()
            logger.info("Metadata tagging completed.")
            
        except Exception as e:
            logger.error(f"Failed to apply Mutagen tags: {str(e)}")
            raise RuntimeError(f"Metadata tagging failed: {str(e)}")