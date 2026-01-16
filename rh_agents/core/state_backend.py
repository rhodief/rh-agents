"""
State Backend Abstraction Layer

Defines abstract interfaces for:
1. StateBackend - Persistence of execution state snapshots
2. ArtifactBackend - Separate storage for large artifacts

This abstraction allows pluggable storage implementations:
- FileSystem (MVP)
- Database (PostgreSQL, SQLite)
- Key-Value stores (Redis, DynamoDB)
- Cloud storage (S3, GCS)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rh_agents.core.state_recovery import StateSnapshot, StateStatus


class StateBackend(ABC):
    """
    Abstract interface for state snapshot persistence.
    
    Implementations must handle:
    - Atomic save/load operations
    - Query by filters (tags, status, pipeline)
    - State lifecycle (create, update, delete)
    """
    
    @abstractmethod
    def save_state(self, snapshot: "StateSnapshot") -> bool:
        """
        Persist a state snapshot.
        
        Args:
            snapshot: StateSnapshot to persist
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Should be atomic - either fully saves or fails cleanly.
            For updates, use update_state() for clarity.
        """
        pass
    
    @abstractmethod
    def load_state(self, state_id: str) -> Optional["StateSnapshot"]:
        """
        Load a state snapshot by its unique ID.
        
        Args:
            state_id: Unique state identifier (UUID)
            
        Returns:
            StateSnapshot if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_states(
        self, 
        tags: Optional[list[str]] = None,
        status: Optional["StateStatus"] = None,
        pipeline_name: Optional[str] = None,
        limit: int = 100
    ) -> list["StateSnapshot"]:
        """
        List states with optional filtering.
        
        Args:
            tags: Filter by tags (states must have ALL specified tags)
            status: Filter by execution status
            pipeline_name: Filter by pipeline name
            limit: Maximum number of results to return
            
        Returns:
            List of matching StateSnapshots, ordered by updated_at (newest first)
            
        Note:
            Implementations should support efficient querying.
            For simple backends (filesystem), in-memory filtering is acceptable.
        """
        pass
    
    @abstractmethod
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a state snapshot.
        
        Args:
            state_id: Unique state identifier to delete
            
        Returns:
            True if state existed and was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def update_state(self, snapshot: "StateSnapshot") -> bool:
        """
        Update an existing state snapshot.
        
        Args:
            snapshot: StateSnapshot with updated data (same state_id)
            
        Returns:
            True if successful, False if state doesn't exist
            
        Note:
            Should update the 'updated_at' timestamp automatically.
        """
        pass
    
    def exists(self, state_id: str) -> bool:
        """
        Check if a state exists without loading it.
        
        Args:
            state_id: Unique state identifier
            
        Returns:
            True if state exists, False otherwise
            
        Note:
            Default implementation uses load_state().
            Subclasses can override for efficiency.
        """
        return self.load_state(state_id) is not None


class ArtifactBackend(ABC):
    """
    Abstract interface for artifact storage (separate from state snapshots).
    
    Artifacts are large objects (embeddings, models, datasets) stored separately
    from state snapshots to keep snapshots lightweight. State snapshots contain
    only artifact references (IDs), not the actual artifacts.
    
    Key design principles:
    - Content-addressable: artifact_id typically = hash(content)
    - Deduplication: Same artifact used in multiple states stored once
    - Lifecycle: Separate from state snapshots (requires garbage collection)
    """
    
    @abstractmethod
    def save_artifact(self, artifact_id: str, artifact: Any) -> bool:
        """
        Store an artifact with the given ID.
        
        Args:
            artifact_id: Unique identifier for the artifact (typically content hash)
            artifact: The artifact object to store
            
        Returns:
            True if successful, False otherwise
            
        Note:
            Implementation should handle serialization (pickle, JSON, etc.)
            based on artifact type and storage backend.
        """
        pass
    
    @abstractmethod
    def load_artifact(self, artifact_id: str) -> Optional[Any]:
        """
        Load an artifact by its ID.
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            The artifact object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            True if artifact existed and was deleted, False if not found
            
        Warning:
            Should only be called after ensuring no states reference this artifact.
            Use garbage collection utilities for safe cleanup.
        """
        pass
    
    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """
        Check if an artifact exists without loading it.
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            True if artifact exists, False otherwise
        """
        pass
    
    def list_artifacts(self) -> list[str]:
        """
        List all artifact IDs stored in this backend.
        
        Returns:
            List of artifact IDs
            
        Note:
            Optional method for backends that support enumeration.
            Used for garbage collection and diagnostics.
            Default implementation returns empty list.
        """
        return []
    
    def get_size(self, artifact_id: str) -> Optional[int]:
        """
        Get the size of an artifact in bytes.
        
        Args:
            artifact_id: Unique artifact identifier
            
        Returns:
            Size in bytes if artifact exists, None otherwise
            
        Note:
            Optional method for storage management and diagnostics.
            Default implementation returns None.
        """
        return None


class StateBackendError(Exception):
    """Base exception for state backend errors"""
    pass


class ArtifactBackendError(Exception):
    """Base exception for artifact backend errors"""
    pass


class StateNotFoundError(StateBackendError):
    """Raised when attempting to load a non-existent state"""
    pass


class ArtifactNotFoundError(ArtifactBackendError):
    """Raised when attempting to load a non-existent artifact"""
    pass


class IncompatibleVersionError(StateBackendError):
    """Raised when state snapshot version is incompatible with current schema"""
    
    def __init__(self, snapshot_version: str, current_version: str):
        self.snapshot_version = snapshot_version
        self.current_version = current_version
        super().__init__(
            f"Incompatible state version: {snapshot_version} (current: {current_version}). "
            f"Migration required."
        )
