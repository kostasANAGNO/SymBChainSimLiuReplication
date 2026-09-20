"""Deterministic roles derived only from an immutable epoch validator set."""

from dataclasses import dataclass

from Chain.ValidatorSet import ValidatorSet


@dataclass(frozen=True, slots=True)
class LiuQuorumRoles:
    client_id: int
    replica_ids: tuple[int, ...]

    @classmethod
    def for_height(cls, validators: ValidatorSet, height: int) -> "LiuQuorumRoles":
        if height <= 0:
            raise ValueError("consensus height must be positive")
        client = validators.proposer_for(height - 1)
        return cls(client, tuple(node_id for node_id in validators.ids if node_id != client))


@dataclass(frozen=True, slots=True)
class LiuPBFTRoles:
    client_id: int
    primary_id: int
    replica_ids: tuple[int, ...]
    backup_ids: tuple[int, ...]

    @classmethod
    def for_height_view(cls, validators: ValidatorSet, height: int, view: int) -> "LiuPBFTRoles":
        if height <= 0:
            raise ValueError("consensus height must be positive")
        if view < 0:
            raise ValueError("PBFT view cannot be negative")
        client = validators.proposer_for(height - 1)
        replicas = tuple(node_id for node_id in validators.ids if node_id != client)
        if not replicas:
            raise ValueError("LiuPBFT requires a client and at least one distinct replica")
        primary = replicas[view % len(replicas)]
        backups = tuple(node_id for node_id in replicas if node_id != primary)
        return cls(client, primary, replicas, backups)


@dataclass(frozen=True, slots=True)
class LiuZyzzyvaRoles:
    """Client/primary/backup mapping shared with the Liu primary-backup abstraction."""

    client_id: int
    primary_id: int
    replica_ids: tuple[int, ...]
    backup_ids: tuple[int, ...]

    @classmethod
    def for_height_view(cls, validators: ValidatorSet, height: int, view: int) -> "LiuZyzzyvaRoles":
        if height <= 0:
            raise ValueError("consensus height must be positive")
        if view < 0:
            raise ValueError("Zyzzyva view cannot be negative")
        client = validators.proposer_for(height)
        replicas = tuple(node_id for node_id in validators.ids if node_id != client)
        if not replicas:
            raise ValueError("LiuZyzzyva requires a client and at least one distinct replica")
        primary = replicas[view % len(replicas)]
        backups = tuple(node_id for node_id in replicas if node_id != primary)
        return cls(client, primary, replicas, backups)
