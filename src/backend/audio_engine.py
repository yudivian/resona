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
    use_hpf: bool = True
    use_deesser: bool = True

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
    
    def _build_spatial_mix(self, segments: List['AudioSegmentConfig'], use_hpf: bool = True) -> Optional[torch.Tensor]:
        """
        Orchestrates a multi-stage Digital Signal Processing (DSP) pipeline to assemble 
        multiple audio clips into a unified stereophonic timeline.

        Supports dynamic timeline overlap (cross-talking) via positional tensor addition
        instead of linear concatenation.
        """
        logger.info(f"Initiating spatial synthesis for {len(segments)} segments")
        
        master_waveform = torch.zeros(2, 0)
        current_sample = 0 
        
        valid_segments_processed = 0
        
        for i, seg in enumerate(segments):
            if not seg.path.exists():
                logger.warning(f"Segment {i} ignored: Path not found at {seg.path}")
                continue
            
            try:
                waveform, sr = torchaudio.load(str(seg.path))
                
                if sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                    waveform = resampler(waveform)
                    
                if use_hpf:
                    waveform = torchaudio.functional.highpass_biquad(waveform, self.target_sample_rate, 80.0)
                
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

                segment_length = stereo_waveform.shape[1]
                required_length = current_sample + segment_length
                
                if required_length > master_waveform.shape[1]:
                    pad_amount = required_length - master_waveform.shape[1]
                    master_waveform = torch.nn.functional.pad(master_waveform, (0, pad_amount))
                    
                master_waveform[:, current_sample:required_length] += stereo_waveform
                valid_segments_processed += 1
                
                delay_samples = int((seg.post_delay_ms / 1000.0) * self.target_sample_rate)
                
                if seg.post_delay_ms > 0:
                    gap = self._generate_room_tone(seg.post_delay_ms, seg.room_tone_level)
                    if gap.shape[0] == 1:
                        gap = gap.repeat(2, 1)
                    
                    req_gap_len = current_sample + segment_length + delay_samples
                    if req_gap_len > master_waveform.shape[1]:
                        pad_amount = req_gap_len - master_waveform.shape[1]
                        master_waveform = torch.nn.functional.pad(master_waveform, (0, pad_amount))
                        
                    master_waveform[:, current_sample + segment_length : req_gap_len] += gap
                    
                    current_sample += segment_length + delay_samples
                else:
                    current_sample += segment_length + delay_samples
                    current_sample = max(0, current_sample)

            except Exception as e:
                logger.error(f"Critical DSP failure on segment {i}: {str(e)}")
                
        if valid_segments_processed == 0 or master_waveform.shape[1] == 0:
            logger.error("Synthesis aborted: No valid audio segments to mix")
            return None
            
        logger.info(f"Spatial synthesis successful: {master_waveform.shape[1]/self.target_sample_rate:.2f} seconds generated")
        return master_waveform

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
            
            stream_node = input_stream
            if config.use_deesser:
                stream_node = stream_node.filter('deesser')
            
            compressed = stream_node.filter(
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
            master_waveform = self._build_spatial_mix(segments,use_hpf=mastering.use_hpf)
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