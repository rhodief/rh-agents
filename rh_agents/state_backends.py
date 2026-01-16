"""
File System Backend Implementations (MVP)

Provides simple file-based storage for:
1. State snapshots (JSON files)
2. Artifacts (pickle files)

Suitable for:
- Local development
- Single-process applications
- Testing
- Small-scale deployments

Limitations:
- No concurrent access control
- No query optimization (loads all states for filtering)
- Limited to local filesystem

For production, consider database or cloud storage backends.
"""
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from rh_agents.core.state_backend import (
    StateBackend, 
    ArtifactBackend,
    StateNotFoundError,
    ArtifactNotFoundError
)
from rh_agents.core.state_recovery import StateSnapshot, StateStatus


class FileSystemStateBackend(StateBackend):
    """
    File-based state storage using JSON.
    
    Directory structure:
        base_path/
            states/
                <state_id>.json
                <state_id>.json
                ...
    
    Features:
    - Human-readable JSON format
    - Simple file-per-state model
    - Atomic writes (write to temp, then rename)
    """
    
    def __init__(self, base_path: str = "./.state_store"):
        """
        Initialize file system state backend.
        
        Args:
            base_path: Base directory for state storage
        """
        self.base_path = Path(base_path)
        self.states_dir = self.base_path / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, snapshot: StateSnapshot) -> bool:
        """
        Save state snapshot as JSON file.
        
        Uses atomic write (write to temp file, then rename) to prevent
        corruption from interrupted writes.
        """
        try:
            file_path = self.states_dir / f"{snapshot.state_id}.json"
            temp_path = self.states_dir / f"{snapshot.state_id}.json.tmp"
            
            # Write to temp file
            with open(temp_path, 'w') as f:
                f.write(snapshot.model_dump_json(indent=2))
            
            # Atomic rename
            temp_path.replace(file_path)
            
            return True
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            print(f"Error saving state {snapshot.state_id}: {e}")
            return False
    
    def load_state(self, state_id: str) -> Optional[StateSnapshot]:
        """Load state snapshot from JSON file."""
        file_path = self.states_dir / f"{state_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return StateSnapshot.model_validate_json(f.read())
        except Exception as e:
            print(f"Error loading state {state_id}: {e}")
            return None
    
    def list_states(
        self,
        tags: Optional[list[str]] = None,
        status: Optional[StateStatus] = None,
        pipeline_name: Optional[str] = None,
        limit: int = 100
    ) -> list[StateSnapshot]:
        """
        List states by loading all and filtering in memory.
        
        Note: Not efficient for large numbers of states.
        Production systems should use database with indexed queries.
        """
        states = []
        
        # Load all state files
        for file_path in self.states_dir.glob("*.json"):
            # Skip temp files
            if file_path.suffix == '.tmp':
                continue
                
            try:
                with open(file_path, 'r') as f:
                    snapshot = StateSnapshot.model_validate_json(f.read())
                    states.append(snapshot)
            except Exception as e:
                print(f"Error loading state file {file_path}: {e}")
                continue
        
        # Apply filters
        filtered_states = []
        for snapshot in states:
            # Filter by tags (must have ALL specified tags)
            if tags and not all(tag in snapshot.metadata.tags for tag in tags):
                continue
            
            # Filter by status
            if status and snapshot.status != status:
                continue
            
            # Filter by pipeline name
            if pipeline_name and snapshot.metadata.pipeline_name != pipeline_name:
                continue
            
            filtered_states.append(snapshot)
        
        # Sort by updated_at (newest first)
        filtered_states.sort(key=lambda s: s.updated_at, reverse=True)
        
        # Apply limit
        return filtered_states[:limit]
    
    def delete_state(self, state_id: str) -> bool:
        """Delete state snapshot file."""
        file_path = self.states_dir / f"{state_id}.json"
        
        if not file_path.exists():
            return False
        
        try:
            file_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting state {state_id}: {e}")
            return False
    
    def update_state(self, snapshot: StateSnapshot) -> bool:
        """
        Update existing state snapshot.
        
        Updates the updated_at timestamp automatically.
        """
        # Check if state exists
        if not self.exists(snapshot.state_id):
            return False
        
        # Update timestamp
        snapshot.updated_at = datetime.now().isoformat()
        
        # Save (overwrites existing file)
        return self.save_state(snapshot)
    
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        state_files = list(self.states_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in state_files)
        
        return {
            "total_states": len(state_files),
            "total_size_bytes": total_size,
            "storage_path": str(self.states_dir)
        }


class FileSystemArtifactBackend(ArtifactBackend):
    """
    File-based artifact storage using pickle.
    
    Directory structure:
        base_path/
            artifacts/
                <artifact_id>.pkl
                <artifact_id>.pkl
                ...
    
    Features:
    - Binary pickle format for Python objects
    - Content-addressable storage (artifact_id = hash)
    - Automatic deduplication
    
    Limitations:
    - Pickle is Python-specific and version-sensitive
    - No type checking on load
    - Security: Don't unpickle untrusted data
    """
    
    def __init__(self, base_path: str = "./.state_store/artifacts"):
        """
        Initialize file system artifact backend.
        
        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_artifact(self, artifact_id: str, artifact: Any) -> bool:
        """
        Save artifact as pickle file.
        
        Uses atomic write to prevent corruption.
        """
        try:
            file_path = self.base_path / f"{artifact_id}.pkl"
            temp_path = self.base_path / f"{artifact_id}.pkl.tmp"
            
            # Write to temp file
            with open(temp_path, 'wb') as f:
                pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.replace(file_path)
            
            return True
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            print(f"Error saving artifact {artifact_id}: {e}")
            return False
    
    def load_artifact(self, artifact_id: str) -> Optional[Any]:
        """Load artifact from pickle file."""
        file_path = self.base_path / f"{artifact_id}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading artifact {artifact_id}: {e}")
            return None
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact file."""
        file_path = self.base_path / f"{artifact_id}.pkl"
        
        if not file_path.exists():
            return False
        
        try:
            file_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting artifact {artifact_id}: {e}")
            return False
    
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact file exists."""
        file_path = self.base_path / f"{artifact_id}.pkl"
        return file_path.exists()
    
    def list_artifacts(self) -> list[str]:
        """List all artifact IDs."""
        return [f.stem for f in self.base_path.glob("*.pkl")]
    
    def get_size(self, artifact_id: str) -> Optional[int]:
        """Get artifact file size in bytes."""
        file_path = self.base_path / f"{artifact_id}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            return file_path.stat().st_size
        except Exception:
            return None
    
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        artifact_files = list(self.base_path.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in artifact_files)
        
        return {
            "total_artifacts": len(artifact_files),
            "total_size_bytes": total_size,
            "storage_path": str(self.base_path)
        }
    
    def garbage_collect(self, referenced_artifact_ids: set[str]) -> int:
        """
        Remove artifacts that are not referenced by any state.
        
        Args:
            referenced_artifact_ids: Set of artifact IDs that are currently in use
            
        Returns:
            Number of artifacts deleted
            
        Warning:
            This is a destructive operation. Ensure referenced_artifact_ids
            is accurate and complete before calling.
        """
        all_artifacts = set(self.list_artifacts())
        orphaned = all_artifacts - referenced_artifact_ids
        
        deleted_count = 0
        for artifact_id in orphaned:
            if self.delete_artifact(artifact_id):
                deleted_count += 1
        
        return deleted_count


def compute_artifact_id(artifact: Any) -> str:
    """
    Compute a unique ID for an artifact based on its content.
    
    Uses SHA256 hash of pickled object for content-addressable storage.
    Falls back to timestamp+random for unpicklable objects.
    
    Args:
        artifact: The artifact object
        
    Returns:
        Unique artifact ID (hex string)
    """
    try:
        # Try to pickle and hash
        serialized = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()
    except Exception:
        # Fallback: timestamp + random + object repr
        import time
        import random
        fallback_str = f"{time.time()}{random.random()}{repr(artifact)}"
        return hashlib.sha256(fallback_str.encode()).hexdigest()
