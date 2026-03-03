"""
Public interface for the Audiobook compiler package.
Orchestrates the entire pipeline: Timing -> Metadata -> Assembly -> Tagging -> Cleanup.
"""

import time
import shutil
import logging
from pathlib import Path

from .schemas import AudiobookManifest, BuildResult
from .timing import TimeCalculator
from .metadata import MetadataWriter
from .assembler import AudiobookAssembler
from .tagger import AudiobookTagger

logger = logging.getLogger("AudiobookBuilder")
logger.setLevel(logging.INFO)

class AudiobookBuilder:
    """
    Facade for compiling an audiobook from a manifest.
    Ensures safe execution and guarantees cleanup of temporary assets.
    """

    @staticmethod
    def compile(manifest: AudiobookManifest, output_dir: Path) -> BuildResult:
        """
        Executes the full compilation pipeline.

        Args:
            manifest (AudiobookManifest): The structured blueprint of the audiobook.
            output_dir (Path): The directory where the final .m4b will be saved.

        Returns:
            BuildResult: A structured summary of the operation's success or failure.
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_title = "".join([c for c in manifest.title if c.isalnum() or c in " -_"]).strip().replace(" ", "_")
        final_output_path = output_dir / f"{safe_title}.m4b"
        
        workspace_dir = output_dir / f".tmp_build_{safe_title}"
        
        try:
            logger.info(f"Initiating audiobook build pipeline for: {manifest.title}")
            
            logger.info("Step 1/4: Calculating timeline...")
            timings = TimeCalculator.calculate_timeline(manifest.chapters)
            
            logger.info("Step 2/4: Generating FFmpeg metadata assets...")
            assets = MetadataWriter.create_assets(manifest, timings, workspace_dir)
            
            logger.info("Step 3/4: Assembling M4B container...")
            assembled_path = AudiobookAssembler.assemble(assets, final_output_path)
            
            logger.info("Step 4/4: Injecting Mutagen tags and cover art...")
            AudiobookTagger.apply_tags(assembled_path, manifest)
            
            execution_time = time.time() - start_time
            logger.info(f"Audiobook compiled successfully in {execution_time:.2f} seconds.")
            
            return BuildResult(
                success=True,
                output_path=final_output_path,
                message="Compilation successful.",
                processing_time_sec=execution_time
            )

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            
            if final_output_path.exists():
                final_output_path.unlink()
                
            return BuildResult(
                success=False,
                output_path=None,
                message=error_msg,
                processing_time_sec=time.time() - start_time
            )

        finally:
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
                logger.debug("Temporary workspace destroyed.")