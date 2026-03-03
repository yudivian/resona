"""
Calculates the exact timeline for the audiobook chapters by reading 
lossless audio headers directly, avoiding high memory consumption.
"""

import wave
from typing import List
from dataclasses import dataclass
from pathlib import Path

from .schemas import Chapter

@dataclass
class ChapterTiming:
    """
    Internal structure holding the calculated timeline for a single chapter.
    Used to pass data cleanly to the metadata writer.
    """
    title: str
    start_ms: int
    end_ms: int
    file_path: Path


class TimeCalculator:
    """
    Engine responsible for measuring audio files and establishing the absolute
    start and end timestamps for the final audiobook container.
    """

    @staticmethod
    def calculate_timeline(chapters: List[Chapter]) -> List[ChapterTiming]:
        """
        Iterates through the provided chapters, calculates their durations in 
        milliseconds using native Python libraries, and builds a sequential timeline.

        Args:
            chapters (List[Chapter]): The ordered list of chapters from the manifest.

        Returns:
            List[ChapterTiming]: The calculated timeline with absolute timestamps.
            
        Raises:
            FileNotFoundError: If a referenced audio file does not exist.
            ValueError: If an audio file is corrupted or not a valid WAV.
        """
        timings: List[ChapterTiming] = []
        current_time_ms: int = 0

        for chapter in chapters:
            if not chapter.audio_path.exists():
                raise FileNotFoundError(f"Missing audio file for chapter '{chapter.title}': {chapter.audio_path}")

            try:
                with wave.open(str(chapter.audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration_ms = int((frames / float(rate)) * 1000)
            except wave.Error as e:
                raise ValueError(f"Corrupted or invalid WAV file at {chapter.audio_path}: {e}")

            start_ms = current_time_ms
            end_ms = current_time_ms + duration_ms

            if chapter.author:
                final_title = f"{chapter.title} - {chapter.author}"
            else:
                final_title = chapter.title

            timings.append(ChapterTiming(
                title=final_title,
                start_ms=start_ms,
                end_ms=end_ms,
                file_path=chapter.audio_path
            ))

            current_time_ms = end_ms

        return timings