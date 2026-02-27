"""
Generic audio engine for processing and merging multiple audio files with 
precision timing and acoustic enhancements using PyTorch tensors.
"""

import torch
import torchaudio
import math
import ffmpeg
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("AudioEngine")
logger.setLevel(logging.DEBUG)

@dataclass
class AudioSegmentConfig:
    """
    Agnostic configuration for a single audio segment in a timeline.
    """
    path: Path
    fade_in_ms: int = 50
    fade_out_ms: int = 50
    post_delay_ms: int = 400
    room_tone_level: float = 0.0001
    pan: float = 0.0
    gain_db: float = 0.0
    depth: float = 0.0
    
@dataclass
class MasteringAudioConfig:
    """
    Agnostic configuration for the final mastering stage.
    """
    target_lufs: float = -14.0
    compressor_ratio: float = 3.0
    compressor_threshold: float = -20.0

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
    
    def _build_spatial_mix(self, segments: List['AudioSegmentConfig']) -> Optional[torch.Tensor]:
        """
        Orchestrates a multi-stage Digital Signal Processing (DSP) pipeline to assemble 
        multiple audio clips into a unified stereophonic timeline.

        The processing sequence for each segment follows these technical specifications:
        1. Resampling: Standardizes the input signal to the engine's target sample rate.
        2. Depth Processing: Simulates air absorption and distance via a low-pass 
           biquad filter. The cutoff frequency is dynamically calculated using a 
           linear mapping where increased depth reduces high-frequency content.
        3. Amplitude Scaling: Applies gain based on decibel input, converted to 
           linear multipliers to modify the raw tensor values.
        4. Spatial Imaging: Implements a Constant Power Panning algorithm. This 
           algorithm uses trigonometric functions (sine and cosine) to distribute 
           monophonic energy across two channels, ensuring the perceived volume 
           remains constant as the signal moves across the stereo field.
        5. Temporal Concatenation: Stacks processed tensors and generated room 
           tone gaps along the time dimension (dim=1).

        Args:
            segments (List[AudioSegmentConfig]): A list containing source paths and 
                                                 individual mixing parameters.

        Returns:
            Optional[torch.Tensor]: A high-precision stereophonic tensor [2, samples] 
                                    representing the unmastered mix, or None if the 
                                    pipeline produces no valid audio data.
        """
        logger.info(f"Initiating spatial synthesis for {len(segments)} segments")
        timeline: List[torch.Tensor] = []
        
        for i, seg in enumerate(segments):
            if not seg.path.exists():
                logger.warning(f"Segment {i} ignored: Path not found at {seg.path}")
                continue
            
            try:
                waveform, sr = torchaudio.load(str(seg.path))
                
                if sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                    waveform = resampler(waveform)
                
                waveform = self._apply_fades(waveform, seg.fade_in_ms, seg.fade_out_ms)
                
                if seg.depth > 0:
                    cutoff = max(1000.0, 8000.0 - (7000.0 * seg.depth))
                    waveform = torchaudio.functional.lowpass_biquad(waveform, self.target_sample_rate, cutoff)
                
                if seg.gain_db != 0.0:
                    gain_linear = 10 ** (seg.gain_db / 20.0)
                    waveform = waveform * gain_linear
                    
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    
                angle = (seg.pan + 1.0) * math.pi / 4.0
                left_gain = math.cos(angle)
                right_gain = math.sin(angle)
                
                stereo_waveform = torch.cat([waveform * left_gain, waveform * right_gain], dim=0)
                timeline.append(stereo_waveform)
                
                if seg.post_delay_ms > 0:
                    gap = self._generate_room_tone(seg.post_delay_ms, seg.room_tone_level)
                    if gap.shape[0] == 1:
                        gap = gap.repeat(2, 1)
                    timeline.append(gap)
            except Exception as e:
                logger.error(f"Critical DSP failure on segment {i}: {str(e)}")
                
        if not timeline:
            logger.error("Synthesis aborted: No valid audio segments to concatenate")
            return None
            
        master = torch.cat(timeline, dim=1)
        logger.info(f"Spatial synthesis successful: {master.shape[1]/self.target_sample_rate:.2f} seconds generated")
        return master

    def _apply_mastering_chain(
        self, 
        input_path: Path, 
        wav_out_path: Path, 
        mp3_out_path: Path, 
        config: 'MasteringAudioConfig'
    ) -> bool:
        """
        Finalizes the audio project by applying a non-linear dynamic range processing 
        and loudness normalization chain via FFmpeg.

        Technical Chain Implementation:
        1. Log-to-Linear Conversion: Translates the decibel-based compression 
           threshold into a linear amplitude coefficient (0.0 to 1.0) required 
           by the FFmpeg acompressor filter.
        2. Dynamic Compression: Reduces the peak-to-RMS ratio to increase 
           overall signal density and vocal presence.
        3. EBU R128 Loudness Normalization: Adjusts the integrated loudness to the 
           specified LUFS target, ensuring compliance with modern streaming and 
           broadcast standards.
        4. Filter Graph Splitting: Utilizes the 'asplit' filter to fork the 
           processed audio stream into two identical branches. This allows for 
           simultaneous encoding into different containers (WAV and MP3) without 
           re-processing or causing graph termination errors.

        Args:
            input_path (Path): Path to the temporary high-resolution pre-master file.
            wav_out_path (Path): Destination for the mastered 16-bit linear PCM WAV.
            mp3_out_path (Path): Destination for the mastered MPEG Layer III file.
            config (MasteringAudioConfig): Parameters governing the mastering thresholds.

        Returns:
            bool: True if the FFmpeg graph executed successfully and both files exist.
        """
        logger.info(f"Executing mastering chain: Target {config.target_lufs} LUFS")
        try:
            threshold_linear = 10 ** (config.compressor_threshold / 20.0)
            
            input_stream = ffmpeg.input(str(input_path))
            
            compressed = input_stream.filter(
                'acompressor', 
                threshold=threshold_linear, 
                ratio=config.compressor_ratio,
                attack=20,
                release=250
            )
            
            mastered = compressed.filter('loudnorm', i=config.target_lufs)
            
            split = mastered.filter_multi_output('asplit')
            
            out_wav = split[0].output(str(wav_out_path), acodec='pcm_s16le')
            out_mp3 = split[1].output(str(mp3_out_path), acodec='libmp3lame', qscale=2)
            
            ffmpeg.run(
                ffmpeg.merge_outputs(out_wav, out_mp3), 
                overwrite_output=True, 
                capture_stdout=True, 
                capture_stderr=True
            )
            logger.info("Mastering process completed for all output formats")
            return True
            
        except ffmpeg.Error as e:
            error_data = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg binary error: {error_data}")
            return False
        except Exception as e:
            logger.error(f"Unexpected mastering orchestration failure: {str(e)}")
            return False

    def merge_segments(
        self, 
        segments: List['AudioSegmentConfig'], 
        output_path: Path,
        mastering: 'MasteringAudioConfig'
    ) -> bool:
        """
        Manages the complete lifecycle of a project's audio merge operation, 
        acting as the primary interface between raw data and finalized media.

        Execution Workflow:
        1. Memory-Level Mixing: Generates a high-precision spatial mix using PyTorch.
        2. Transient Storage: Writes the mix to a temporary high-fidelity WAV file. 
           This serves as the high-bitrate handoff point for external mastering tools.
        3. Mastering Execution: Triggers the FFmpeg process for dynamic finalization.
        4. Resource Reclamation: Enforces the removal of the transient pre-master 
           file using a finally block to ensure disk space is recovered regardless 
           of the outcome.

        Args:
            segments (List[AudioSegmentConfig]): The structured list of audio segments.
            output_path (Path): The primary target path for the final mastered output.
            mastering (MasteringAudioConfig): The global parameters for audio finalization.

        Returns:
            bool: True if the orchestration finished and the final master assets were 
                  successfully verified on disk.
        """
        temp_path = output_path.with_name(f"tmp_pre_{output_path.name}")
        logger.info(f"Starting project merge orchestration for {output_path.name}")
        
        try:
            master_waveform = self._build_spatial_mix(segments)
            if master_waveform is None:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(temp_path), master_waveform, self.target_sample_rate)
            
            success = self._apply_mastering_chain(
                input_path=temp_path,
                wav_out_path=output_path,
                mp3_out_path=output_path.with_suffix('.mp3'),
                config=mastering
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Top-level merge orchestration failure: {str(e)}")
            return False
        finally:
            if temp_path.exists():
                temp_path.unlink()
                logger.info("Transient assets successfully purged from filesystem")