"""
Generic audio engine for processing and merging multiple audio files with 
precision timing and acoustic enhancements using PyTorch tensors.
"""

import torch
import torchaudio
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AudioSegmentConfig:
    """
    Agostic configuration for a single audio segment in a timeline.
    """
    path: Path
    fade_in_ms: int = 50
    fade_out_ms: int = 50
    post_delay_ms: int = 400
    room_tone_level: float = 0.0001

class AudioEngine:
    """
    Generic high-performance audio processor. 
    
    Operates independently of any application-specific models, receiving 
    raw configuration objects and outputting processed master files.
    """

    def __init__(self, target_sample_rate: int = 24000):
        """
        Initializes the engine with a consistent sample rate.

        Args:
            target_sample_rate (int): Frequency to which all segments 
                                      will be synchronized.
        """
        self.target_sample_rate = target_sample_rate

    def _apply_fades(self, waveform: torch.Tensor, fade_in_ms: int, fade_out_ms: int) -> torch.Tensor:
        """
        Applies linear fades to eliminate digital clicks at segment boundaries.

        Args:
            waveform (torch.Tensor): The audio tensor to process.
            fade_in_ms (int): Duration of the start fade.
            fade_out_ms (int): Duration of the end fade.

        Returns:
            torch.Tensor: Faded waveform.
        """
        num_channels, num_frames = waveform.shape
        
        if fade_in_ms > 0:
            fade_in_frames = min(int((fade_in_ms / 1000.0) * self.target_sample_rate), num_frames)
            fade_in_curve = torch.linspace(0.0, 1.0, fade_in_frames)
            waveform[:, :fade_in_frames] *= fade_in_curve
            
        if fade_out_ms > 0:
            fade_out_frames = min(int((fade_out_ms / 1000.0) * self.target_sample_rate), num_frames)
            fade_out_curve = torch.linspace(1.0, 0.0, fade_out_frames)
            waveform[:, num_frames - fade_out_frames:] *= fade_out_curve
            
        return waveform

    def _generate_room_tone(self, duration_ms: int, level: float) -> torch.Tensor:
        """
        Generates low-level noise to maintain acoustic presence during gaps.

        Args:
            duration_ms (int): Gap length in milliseconds.
            level (float): Noise floor amplitude.

        Returns:
            torch.Tensor: Synthetic ambient noise.
        """
        if duration_ms <= 0:
            return torch.empty(1, 0)
            
        num_frames = int((duration_ms / 1000.0) * self.target_sample_rate)
        noise = torch.randn(1, num_frames) * level
        return noise

    # def merge_segments(self, segments: List[AudioSegmentConfig], output_path: Path) -> bool:
    #     """
    #     Merges a sequence of audio segments into a single master file.
    #     This method is completely agnostic to application-level logic.

    #     Args:
    #         segments (List[AudioSegmentConfig]): Ordered list of audio configurations.
    #         output_path (Path): Final destination for the merged WAV file.

    #     Returns:
    #         bool: True if the file was successfully written, False otherwise.
    #     """
    #     timeline: List[torch.Tensor] = []
        
    #     try:
    #         for seg in segments:
    #             if not seg.path.exists():
    #                 continue
                
    #             waveform, sr = torchaudio.load(str(seg.path))
                
    #             if sr != self.target_sample_rate:
    #                 resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
    #                 waveform = resampler(waveform)
                
    #             waveform = self._apply_fades(waveform, seg.fade_in_ms, seg.fade_out_ms)
    #             timeline.append(waveform)
                
    #             if seg.post_delay_ms > 0:
    #                 gap = self._generate_room_tone(seg.post_delay_ms, seg.room_tone_level)
    #                 timeline.append(gap)
            
    #         if not timeline:
    #             return False
                
    #         master_waveform = torch.cat(timeline, dim=1)
            
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         torchaudio.save(str(output_path), master_waveform, self.target_sample_rate)
            
    #         return True
            
    #     except Exception:
    #         return False
    
    def merge_segments(self, segments: List['AudioSegmentConfig'], output_path: Path) -> bool:
        """
        Concatenates an ordered list of audio segments into a single continuous 
        master timeline, exporting both a monaural WAV file and a stereo MP3 file.

        Args:
            segments (List[AudioSegmentConfig]): Ordered list of audio configurations.
            output_path (Path): Final destination for the merged WAV file. The MP3 
                                equivalent will be saved in the same directory.

        Returns:
            bool: True if both files were successfully written, False otherwise.
        """
        timeline: List[torch.Tensor] = []
        
        try:
            for seg in segments:
                if not seg.path.exists():
                    continue
                
                waveform, sr = torchaudio.load(str(seg.path))
                
                if sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                    waveform = resampler(waveform)
                
                waveform = self._apply_fades(waveform, seg.fade_in_ms, seg.fade_out_ms)
                timeline.append(waveform)
                
                if seg.post_delay_ms > 0:
                    gap = self._generate_room_tone(seg.post_delay_ms, seg.room_tone_level)
                    timeline.append(gap)
            
            if not timeline:
                return False
                
            master_waveform = torch.cat(timeline, dim=1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torchaudio.save(str(output_path), master_waveform, self.target_sample_rate)
            
            stereo_waveform = master_waveform.repeat(2, 1)
            mp3_path = output_path.with_suffix('.mp3')
            
            torchaudio.save(str(mp3_path), stereo_waveform, self.target_sample_rate)
            
            return True
            
        except Exception:
            return False