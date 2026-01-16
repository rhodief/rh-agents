"""
Test Phase 1: State Recovery Foundation

Tests basic state save/restore functionality including:
- Creating state snapshots
- Saving to file system backend
- Loading from file system backend
- Artifact storage and recovery
"""
import asyncio
import pytest
from pathlib import Path
import shutil

from rh_agents.core.execution import ExecutionState, ExecutionStore
from rh_agents.core.state_recovery import StateSnapshot, StateStatus, StateMetadata, ReplayMode
from rh_agents.state_backends import FileSystemStateBackend, FileSystemArtifactBackend


def test_state_snapshot_creation():
    """Test creating a state snapshot."""
    # Create execution state
    state = ExecutionState()
    state.storage.set("test_key", "test_value")
    state.execution_stack.append("agent1")
    
    # Create snapshot
    snapshot = state.to_snapshot(
        status=StateStatus.RUNNING,
        metadata=StateMetadata(
            tags=["test"],
            description="Test snapshot",
            pipeline_name="test_pipeline"
        )
    )
    
    # Verify snapshot
    assert snapshot.state_id == state.state_id
    assert snapshot.status == StateStatus.RUNNING
    assert snapshot.metadata.tags == ["test"]
    assert snapshot.metadata.pipeline_name == "test_pipeline"
    assert "storage" in snapshot.execution_state
    assert "execution_stack" in snapshot.execution_state
    print("✅ State snapshot creation successful")


def test_filesystem_state_backend():
    """Test file system state backend save/load."""
    test_dir = Path("./.test_state_store")
    
    # Clean up any previous test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backend
        backend = FileSystemStateBackend(str(test_dir))
        
        # Create state and snapshot
        state = ExecutionState()
        state.storage.set("key1", "value1")
        state.storage.set("key2", "value2")
        snapshot = state.to_snapshot(status=StateStatus.RUNNING)
        
        # Save snapshot
        success = backend.save_state(snapshot)
        assert success, "Failed to save state"
        print(f"✅ Saved state {snapshot.state_id}")
        
        # Load snapshot
        loaded_snapshot = backend.load_state(snapshot.state_id)
        assert loaded_snapshot is not None, "Failed to load state"
        assert loaded_snapshot.state_id == snapshot.state_id
        assert loaded_snapshot.execution_state["storage"]["data"]["key1"] == "value1"
        print(f"✅ Loaded state {loaded_snapshot.state_id}")
        
        # Test list_states
        states = backend.list_states()
        assert len(states) == 1
        assert states[0].state_id == snapshot.state_id
        print(f"✅ Listed {len(states)} states")
        
        # Test delete
        deleted = backend.delete_state(snapshot.state_id)
        assert deleted, "Failed to delete state"
        assert backend.load_state(snapshot.state_id) is None
        print(f"✅ Deleted state {snapshot.state_id}")
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_filesystem_artifact_backend():
    """Test file system artifact backend."""
    test_dir = Path("./.test_state_store/artifacts")
    
    # Clean up any previous test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backend
        backend = FileSystemArtifactBackend(str(test_dir))
        
        # Test data
        test_artifact = {"embedding": [1.0, 2.0, 3.0], "metadata": "test"}
        artifact_id = "test_artifact_123"
        
        # Save artifact
        success = backend.save_artifact(artifact_id, test_artifact)
        assert success, "Failed to save artifact"
        print(f"✅ Saved artifact {artifact_id}")
        
        # Load artifact
        loaded = backend.load_artifact(artifact_id)
        assert loaded is not None, "Failed to load artifact"
        assert loaded["embedding"] == [1.0, 2.0, 3.0]
        print(f"✅ Loaded artifact {artifact_id}")
        
        # Test exists
        assert backend.exists(artifact_id)
        print(f"✅ Artifact exists check passed")
        
        # Test list
        artifacts = backend.list_artifacts()
        assert artifact_id in artifacts
        print(f"✅ Listed {len(artifacts)} artifacts")
        
        # Test delete
        deleted = backend.delete_artifact(artifact_id)
        assert deleted
        assert not backend.exists(artifact_id)
        print(f"✅ Deleted artifact {artifact_id}")
        
    finally:
        # Clean up
        if test_dir.parent.exists():
            shutil.rmtree(test_dir.parent)


def test_state_restoration():
    """Test full state save and restore cycle."""
    test_dir = Path("./.test_state_store")
    
    # Clean up any previous test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    try:
        # Create backends
        state_backend = FileSystemStateBackend(str(test_dir))
        artifact_backend = FileSystemArtifactBackend(str(test_dir / "artifacts"))
        
        # Create original state
        original_state = ExecutionState(
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        original_state.storage.set("step1", "completed")
        original_state.storage.set("step2", "in_progress")
        original_state.storage.set_artifact("embedding", [1.0, 2.0, 3.0])
        original_state.execution_stack.append("agent1")
        original_state.execution_stack.append("tool1")
        
        print(f"✅ Created original state {original_state.state_id}")
        
        # Save checkpoint
        success = original_state.save_checkpoint(
            status=StateStatus.PAUSED,
            metadata=StateMetadata(
                tags=["test", "checkpoint"],
                description="Test checkpoint",
                pipeline_name="test_pipeline"
            )
        )
        assert success, "Failed to save checkpoint"
        print(f"✅ Saved checkpoint")
        
        # Restore state
        restored_state = ExecutionState.load_from_state_id(
            state_id=original_state.state_id,
            state_backend=state_backend,
            artifact_backend=artifact_backend
        )
        
        assert restored_state is not None, "Failed to restore state"
        print(f"✅ Restored state {restored_state.state_id}")
        
        # Verify restored state matches original
        assert restored_state.state_id == original_state.state_id
        assert restored_state.storage.get("step1") == "completed"
        assert restored_state.storage.get("step2") == "in_progress"
        assert restored_state.storage.get_artifact("embedding") == [1.0, 2.0, 3.0]
        assert restored_state.execution_stack == ["agent1", "tool1"]
        print(f"✅ Verified restored state matches original")
        
        # Test metadata filtering
        states = state_backend.list_states(tags=["test"])
        assert len(states) == 1
        assert states[0].metadata.pipeline_name == "test_pipeline"
        print(f"✅ Metadata filtering works")
        
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_state_diff():
    """Test state diff functionality."""
    from rh_agents.core.state_recovery import StateSnapshot
    
    # Create two states
    state1 = ExecutionState()
    state1.storage.set("key1", "value1")
    state1.storage.set_artifact("art1", {"data": "old"})
    snapshot1 = state1.to_snapshot()
    
    state2 = ExecutionState()
    state2.storage.set("key1", "value1_modified")
    state2.storage.set("key2", "value2")
    state2.storage.set_artifact("art1", {"data": "old"})
    state2.storage.set_artifact("art2", {"data": "new"})
    snapshot2 = state2.to_snapshot()
    
    # Compute diff
    diff = StateSnapshot.diff(snapshot1, snapshot2)
    
    # Verify diff
    assert "key1" in diff.changed_storage
    assert diff.changed_storage["key1"] == ("value1", "value1_modified")
    assert "key2" in diff.changed_storage
    assert "art2" in diff.new_artifacts
    print(f"✅ State diff computed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 1 TESTS: State Recovery Foundation")
    print("="*60 + "\n")
    
    print("Test 1: State Snapshot Creation")
    print("-" * 40)
    test_state_snapshot_creation()
    
    print("\nTest 2: FileSystem State Backend")
    print("-" * 40)
    test_filesystem_state_backend()
    
    print("\nTest 3: FileSystem Artifact Backend")
    print("-" * 40)
    test_filesystem_artifact_backend()
    
    print("\nTest 4: Full State Restoration")
    print("-" * 40)
    test_state_restoration()
    
    print("\nTest 5: State Diff")
    print("-" * 40)
    test_state_diff()
    
    print("\n" + "="*60)
    print("✅ ALL PHASE 1 TESTS PASSED!")
    print("="*60 + "\n")
