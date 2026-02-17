import io
import json
import zipfile
import numpy as np
from typing import Tuple, Dict, Any
from src.models import VoiceProfile

class VoiceBundleIO:
    """
    Service responsible for the serialization and de-serialization of Resona Voice Bundles.
    
    A Resona Bundle (.rnb) is a ZIP-compressed package containing:
    1. profile.json: Metadata, tags, and source information.
    2. identity.npy: The raw embedding vector representing the voice's identity.
    3. anchor.wav: The original reference audio file.
    """

    @staticmethod
    def pack_bundle(profile: VoiceProfile, identity_vector: list, anchor_audio_bytes: bytes) -> bytes:
        """
        Creates a compressed Resona Bundle in memory from voice components.

        Args:
            profile (VoiceProfile): The metadata object of the voice.
            identity_vector (list): The raw list of floats representing the embedding.
            anchor_audio_bytes (bytes): The binary content of the reference audio file.

        Returns:
            bytes: The binary content of the resulting .rnb (ZIP) file.
        """
        buffer = io.BytesIO()
        
        profile_data = profile.model_dump()
        # Ensure semantic embedding is excluded if present to keep bundles portable and focused on identity
        if "semantic_embedding" in profile_data:
            profile_data["semantic_embedding"] = None

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Add Profile Metadata
            zf.writestr("profile.json", json.dumps(profile_data, indent=4))
            
            # 2. Add Identity Vector (Compressed NPY)
            vector_array = np.array(identity_vector, dtype=np.float32)
            vector_buffer = io.BytesIO()
            np.save(vector_buffer, vector_array)
            zf.writestr("identity.npy", vector_buffer.getvalue())
            
            # 3. Add Anchor Audio
            zf.writestr("anchor.wav", anchor_audio_bytes)
            
        return buffer.getvalue()

    @staticmethod
    def unpack_bundle(bundle_bytes: bytes) -> Tuple[Dict[str, Any], list, bytes]:
        """
        Extracts voice components from a binary Resona Bundle.

        Args:
            bundle_bytes (bytes): The binary content of the .rnb file.

        Returns:
            Tuple[Dict[str, Any], list, bytes]: A tuple containing:
                - The profile dictionary (to be converted back to VoiceProfile).
                - The identity vector as a list of floats.
                - The raw binary content of the anchor audio.

        Raises:
            ValueError: If the bundle is corrupted or missing essential components.
        """
        buffer = io.BytesIO(bundle_bytes)
        
        if not zipfile.is_zipfile(buffer):
            raise ValueError("Invalid bundle format: Not a Resona Bundle or ZIP file.")

        with zipfile.ZipFile(buffer, 'r') as zf:
            required_files = ["profile.json", "identity.npy", "anchor.wav"]
            for f in required_files:
                if f not in zf.namelist():
                    raise ValueError(f"Corrupted bundle: Missing {f}")

            # 1. Extract Profile
            profile_dict = json.loads(zf.read("profile.json"))
            
            # 2. Extract Identity Vector
            vector_buffer = io.BytesIO(zf.read("identity.npy"))
            vector_array = np.load(vector_buffer)
            identity_vector = vector_array.tolist()
            
            # 3. Extract Anchor Audio
            anchor_audio_bytes = zf.read("anchor.wav")

        return profile_dict, identity_vector, anchor_audio_bytes