"""Immutable certificate evidence shared by Liu runtime protocols."""

from dataclasses import dataclass

from Liu.Serialization import CanonicalSerializable
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity


def _normalized_quorum_signers(signer_ids: tuple[int, ...], threshold: int) -> tuple[int, ...]:
    signers = tuple(sorted(signer_ids))
    if threshold <= 0:
        raise ValueError("certificate threshold must be positive")
    if len(signers) != len(set(signers)):
        raise ValueError("certificate signer IDs must be distinct")
    if len(signers) != threshold:
        raise ValueError("certificate must contain exactly threshold distinct signers")
    return signers


@dataclass(frozen=True, slots=True)
class PBFTPreparedCertificate(CanonicalSerializable):
    epoch_id: int
    height: int
    view: int
    block_digest: str
    parent_digest: str
    signer_ids: tuple[int, ...]
    threshold: int
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer_ids", _normalized_quorum_signers(self.signer_ids, self.threshold))

    def validate(self, identity: LiuBlockIdentity, view: int, replica_ids: tuple[int, ...], threshold: int) -> bool:
        return (
            self.epoch_id == identity.epoch_id
            and self.height == identity.height
            and self.view == view
            and self.block_digest == identity.block_digest
            and self.parent_digest == identity.parent_digest
            and self.threshold == threshold
            and len(self.signer_ids) == threshold
            and set(self.signer_ids).issubset(replica_ids)
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "parent_digest": self.parent_digest,
            "signer_ids": list(self.signer_ids),
            "threshold": self.threshold,
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class PBFTCommitCertificate(CanonicalSerializable):
    epoch_id: int
    height: int
    view: int
    block_digest: str
    parent_digest: str
    signer_ids: tuple[int, ...]
    threshold: int
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer_ids", _normalized_quorum_signers(self.signer_ids, self.threshold))

    def validate(self, identity: LiuBlockIdentity, view: int, replica_ids: tuple[int, ...], threshold: int) -> bool:
        return (
            self.epoch_id == identity.epoch_id
            and self.height == identity.height
            and self.view == view
            and self.block_digest == identity.block_digest
            and self.parent_digest == identity.parent_digest
            and self.threshold == threshold
            and len(self.signer_ids) == threshold
            and set(self.signer_ids).issubset(replica_ids)
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "parent_digest": self.parent_digest,
            "signer_ids": list(self.signer_ids),
            "threshold": self.threshold,
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class PBFTFinalityCertificate(CanonicalSerializable):
    block_identity: LiuBlockIdentity
    view: int
    client_id: int
    reply_evidence: tuple[tuple[int, str], ...]
    threshold: int
    created_at: float

    def __post_init__(self) -> None:
        evidence = tuple(sorted(self.reply_evidence))
        signer_ids = tuple(signer_id for signer_id, _ in evidence)
        if len(evidence) != self.threshold or len(signer_ids) != len(set(signer_ids)):
            raise ValueError("PBFT finality certificate requires exactly threshold distinct reply signers")
        if any(not certificate_hash for _, certificate_hash in evidence):
            raise ValueError("PBFT finality evidence requires non-empty commit-certificate hashes")
        object.__setattr__(self, "reply_evidence", evidence)

    @property
    def signer_ids(self) -> tuple[int, ...]:
        return tuple(signer_id for signer_id, _ in self.reply_evidence)

    def validate(
        self,
        client_id: int,
        replica_ids: tuple[int, ...],
        identity: LiuBlockIdentity,
        reply_threshold: int,
        commit_certificates: tuple[tuple[int, PBFTCommitCertificate], ...],
        commit_threshold: int,
    ) -> bool:
        supplied = {signer_id: certificate for signer_id, certificate in commit_certificates}
        if len(supplied) != len(commit_certificates):
            return False
        return (
            self.client_id == client_id
            and self.block_identity == identity
            and self.threshold == reply_threshold
            and len(self.reply_evidence) == reply_threshold
            and set(self.signer_ids).issubset(replica_ids)
            and set(supplied) == set(self.signer_ids)
            and all(
                supplied[signer_id].deterministic_hash() == certificate_hash
                and supplied[signer_id].validate(identity, self.view, replica_ids, commit_threshold)
                for signer_id, certificate_hash in self.reply_evidence
            )
        )

    def to_dict(self) -> dict:
        return {
            "block_identity": self.block_identity.to_dict(),
            "client_id": self.client_id,
            "created_at": self.created_at,
            "reply_evidence": [
                {"commit_certificate_hash": certificate_hash, "signer_id": signer_id}
                for signer_id, certificate_hash in self.reply_evidence
            ],
            "threshold": self.threshold,
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class PBFTViewChangeEvidence(CanonicalSerializable):
    epoch_id: int
    height: int
    target_view: int
    sender_id: int
    highest_prepared_certificate: PBFTPreparedCertificate | None
    local_commit_certificate: PBFTCommitCertificate | None

    def validate(self, replica_ids: tuple[int, ...], prepare_threshold: int, commit_threshold: int) -> bool:
        if self.target_view <= 0 or self.sender_id not in replica_ids:
            return False
        if self.highest_prepared_certificate is not None:
            certificate = self.highest_prepared_certificate
            identity = LiuBlockIdentity(self.epoch_id, self.height, certificate.parent_digest, certificate.block_digest)
            if certificate.view >= self.target_view or not certificate.validate(
                identity, certificate.view, replica_ids, prepare_threshold
            ):
                return False
        if self.local_commit_certificate is not None:
            certificate = self.local_commit_certificate
            identity = LiuBlockIdentity(self.epoch_id, self.height, certificate.parent_digest, certificate.block_digest)
            if certificate.view >= self.target_view or not certificate.validate(
                identity, certificate.view, replica_ids, commit_threshold
            ):
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "height": self.height,
            "highest_prepared_certificate": (
                None if self.highest_prepared_certificate is None else self.highest_prepared_certificate.to_dict()
            ),
            "local_commit_certificate": (
                None if self.local_commit_certificate is None else self.local_commit_certificate.to_dict()
            ),
            "sender_id": self.sender_id,
            "target_view": self.target_view,
        }


def select_pbft_safe_prepared(
    evidence: tuple[PBFTViewChangeEvidence, ...],
) -> tuple[PBFTPreparedCertificate | None, str | None]:
    committed = [item.local_commit_certificate for item in evidence if item.local_commit_certificate is not None]
    committed_digests = {certificate.block_digest for certificate in committed}
    if len(committed_digests) > 1:
        return None, "conflicting_local_commit_certificates"
    prepared = [item.highest_prepared_certificate for item in evidence if item.highest_prepared_certificate is not None]
    if not prepared:
        if committed:
            return None, "commit_without_prepared_evidence"
        return None, None
    highest_view = max(certificate.view for certificate in prepared)
    highest = [certificate for certificate in prepared if certificate.view == highest_view]
    highest_digests = {certificate.block_digest for certificate in highest}
    if len(highest_digests) != 1:
        return None, "conflicting_highest_view_prepared_certificates"
    selected = min(highest, key=lambda certificate: certificate.deterministic_hash())
    if committed_digests and selected.block_digest not in committed_digests:
        return None, "prepared_value_conflicts_with_local_commit"
    return selected, None


@dataclass(frozen=True, slots=True)
class PBFTNewViewCertificate(CanonicalSerializable):
    QUORUM_POLICY_VERSION = "liu_pbft_view_change_quorum_v1"
    SAFE_VALUE_POLICY_VERSION = "liu_pbft_safe_value_selection_v1"

    epoch_id: int
    height: int
    target_view: int
    evidence: tuple[PBFTViewChangeEvidence, ...]
    threshold: int
    selected_prepared_certificate: PBFTPreparedCertificate | None
    created_at: float

    def __post_init__(self) -> None:
        evidence = tuple(sorted(self.evidence, key=lambda item: item.sender_id))
        signer_ids = tuple(item.sender_id for item in evidence)
        if self.target_view <= 0:
            raise ValueError("NEW_VIEW target must be positive")
        if len(evidence) != self.threshold or len(signer_ids) != len(set(signer_ids)):
            raise ValueError("NEW_VIEW certificate requires exactly threshold distinct evidence signers")
        selected, error = select_pbft_safe_prepared(evidence)
        if error is not None:
            raise ValueError(error)
        if selected != self.selected_prepared_certificate:
            raise ValueError("selected prepared certificate does not follow the safe-value policy")
        object.__setattr__(self, "evidence", evidence)

    @property
    def signer_ids(self) -> tuple[int, ...]:
        return tuple(item.sender_id for item in self.evidence)

    def validate(
        self,
        replica_ids: tuple[int, ...],
        threshold: int,
        prepare_threshold: int,
        commit_threshold: int,
    ) -> bool:
        if self.threshold != threshold or len(self.evidence) != threshold:
            return False
        if not set(self.signer_ids).issubset(replica_ids):
            return False
        if any(
            evidence.epoch_id != self.epoch_id
            or evidence.height != self.height
            or evidence.target_view != self.target_view
            or not evidence.validate(replica_ids, prepare_threshold, commit_threshold)
            for evidence in self.evidence
        ):
            return False
        selected, error = select_pbft_safe_prepared(self.evidence)
        return error is None and selected == self.selected_prepared_certificate

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "height": self.height,
            "quorum_policy": self.QUORUM_POLICY_VERSION,
            "safe_value_policy": self.SAFE_VALUE_POLICY_VERSION,
            "selected_prepared_certificate": (
                None if self.selected_prepared_certificate is None else self.selected_prepared_certificate.to_dict()
            ),
            "target_view": self.target_view,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class ZyzzyvaFastReplyCertificate(CanonicalSerializable):
    epoch_id: int
    height: int
    view: int
    block_digest: str
    parent_digest: str
    signer_ids: tuple[int, ...]
    threshold: int
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer_ids", _normalized_quorum_signers(self.signer_ids, self.threshold))

    def validate(
        self,
        identity: LiuBlockIdentity,
        view: int,
        expected_replica_ids: tuple[int, ...],
    ) -> bool:
        expected = tuple(sorted(expected_replica_ids))
        return (
            self.epoch_id == identity.epoch_id
            and self.height == identity.height
            and self.view == view
            and self.block_digest == identity.block_digest
            and self.parent_digest == identity.parent_digest
            and self.threshold == len(expected)
            and self.signer_ids == expected
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "parent_digest": self.parent_digest,
            "signer_ids": list(self.signer_ids),
            "threshold": self.threshold,
            "view": self.view,
        }


def _normalized_at_least_quorum_signers(signer_ids: tuple[int, ...], threshold: int) -> tuple[int, ...]:
    signers = tuple(sorted(signer_ids))
    if threshold <= 0:
        raise ValueError("certificate threshold must be positive")
    if len(signers) != len(set(signers)):
        raise ValueError("certificate signer IDs must be distinct")
    if len(signers) < threshold:
        raise ValueError("certificate must contain at least threshold distinct signers")
    return signers


@dataclass(frozen=True, slots=True)
class ZyzzyvaCommitCertificate(CanonicalSerializable):
    QUORUM_POLICY_VERSION = "liu_zyzzyva_recovery_quorum_v1"
    SAFE_DIGEST_POLICY_VERSION = "liu_zyzzyva_recovery_safe_digest_v1"

    epoch_id: int
    height: int
    view: int
    block_digest: str
    parent_digest: str
    speculative_reply_signer_ids: tuple[int, ...]
    threshold: int
    creation_time: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "speculative_reply_signer_ids",
            _normalized_at_least_quorum_signers(self.speculative_reply_signer_ids, self.threshold),
        )

    @property
    def signer_ids(self) -> tuple[int, ...]:
        return self.speculative_reply_signer_ids

    def validate(
        self,
        identity: LiuBlockIdentity,
        view: int,
        replica_ids: tuple[int, ...],
        threshold: int,
    ) -> bool:
        return (
            self.epoch_id == identity.epoch_id
            and self.height == identity.height
            and self.view == view
            and self.block_digest == identity.block_digest
            and self.parent_digest == identity.parent_digest
            and self.threshold == threshold
            and len(self.speculative_reply_signer_ids) >= threshold
            and set(self.speculative_reply_signer_ids).issubset(replica_ids)
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "creation_time": self.creation_time,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "parent_digest": self.parent_digest,
            "quorum_policy": self.QUORUM_POLICY_VERSION,
            "safe_digest_policy": self.SAFE_DIGEST_POLICY_VERSION,
            "speculative_reply_signer_ids": list(self.speculative_reply_signer_ids),
            "threshold": self.threshold,
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class ZyzzyvaRecoveryFinalityCertificate(CanonicalSerializable):
    QUORUM_POLICY_VERSION = "liu_zyzzyva_recovery_quorum_v1"

    epoch_id: int
    height: int
    view: int
    block_digest: str
    parent_digest: str
    client_id: int
    local_commit_signer_ids: tuple[int, ...]
    threshold: int
    commit_certificate_hash: str
    creation_time: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_commit_signer_ids",
            _normalized_at_least_quorum_signers(self.local_commit_signer_ids, self.threshold),
        )
        if not self.commit_certificate_hash:
            raise ValueError("recovery finality requires a commit-certificate hash")

    @property
    def signer_ids(self) -> tuple[int, ...]:
        return self.local_commit_signer_ids

    def validate(
        self,
        identity: LiuBlockIdentity,
        view: int,
        client_id: int,
        replica_ids: tuple[int, ...],
        threshold: int,
        commit_certificate: ZyzzyvaCommitCertificate,
    ) -> bool:
        return (
            self.epoch_id == identity.epoch_id
            and self.height == identity.height
            and self.view == view
            and self.block_digest == identity.block_digest
            and self.parent_digest == identity.parent_digest
            and self.client_id == client_id
            and self.threshold == threshold
            and len(self.local_commit_signer_ids) >= threshold
            and set(self.local_commit_signer_ids).issubset(replica_ids)
            and self.commit_certificate_hash == commit_certificate.deterministic_hash()
            and commit_certificate.view <= view
            and commit_certificate.validate(identity, commit_certificate.view, replica_ids, threshold)
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "client_id": self.client_id,
            "commit_certificate_hash": self.commit_certificate_hash,
            "creation_time": self.creation_time,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "local_commit_signer_ids": list(self.local_commit_signer_ids),
            "parent_digest": self.parent_digest,
            "quorum_policy": self.QUORUM_POLICY_VERSION,
            "threshold": self.threshold,
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class ZyzzyvaSpeculativeEvidence(CanonicalSerializable):
    """Matching speculative witnesses for one digest in one originating view."""

    epoch_id: int
    height: int
    originating_view: int
    block_digest: str
    parent_digest: str
    signer_ids: tuple[int, ...]
    threshold: int
    creation_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer_ids", _normalized_at_least_quorum_signers(self.signer_ids, self.threshold))

    def validate(self, replica_ids: tuple[int, ...]) -> bool:
        return (
            self.epoch_id >= 0
            and self.height > 0
            and self.originating_view >= 0
            and bool(self.block_digest)
            and bool(self.parent_digest)
            and set(self.signer_ids).issubset(replica_ids)
        )

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "creation_time": self.creation_time,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "originating_view": self.originating_view,
            "parent_digest": self.parent_digest,
            "signer_ids": list(self.signer_ids),
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class ZyzzyvaViewChangeEvidence(CanonicalSerializable):
    epoch_id: int
    height: int
    target_view: int
    sender_id: int
    highest_speculative_evidence: ZyzzyvaSpeculativeEvidence | None
    commit_certificate: ZyzzyvaCommitCertificate | None
    local_commit_digest: str | None
    locked_digest: str | None

    def validate(self, replica_ids: tuple[int, ...], recovery_threshold: int) -> bool:
        if self.target_view <= 0 or self.sender_id not in replica_ids:
            return False
        speculative = self.highest_speculative_evidence
        if speculative is not None:
            if (
                speculative.epoch_id != self.epoch_id
                or speculative.height != self.height
                or speculative.originating_view >= self.target_view
                or not speculative.validate(replica_ids)
            ):
                return False
        commit = self.commit_certificate
        if commit is not None:
            identity = LiuBlockIdentity(self.epoch_id, self.height, commit.parent_digest, commit.block_digest)
            if commit.view >= self.target_view or not commit.validate(identity, commit.view, replica_ids, recovery_threshold):
                return False
        if self.local_commit_digest is not None:
            if commit is None or self.local_commit_digest != commit.block_digest:
                return False
        if self.locked_digest is not None:
            if commit is None or self.locked_digest != commit.block_digest:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "commit_certificate": None if self.commit_certificate is None else self.commit_certificate.to_dict(),
            "epoch_id": self.epoch_id,
            "height": self.height,
            "highest_speculative_evidence": (
                None if self.highest_speculative_evidence is None else self.highest_speculative_evidence.to_dict()
            ),
            "local_commit_digest": self.local_commit_digest,
            "locked_digest": self.locked_digest,
            "sender_id": self.sender_id,
            "target_view": self.target_view,
        }


def select_zyzzyva_safe_value(
    evidence: tuple[ZyzzyvaViewChangeEvidence, ...],
    speculative_threshold: int,
) -> tuple[ZyzzyvaCommitCertificate | None, ZyzzyvaSpeculativeEvidence | None, str | None]:
    """Select commit evidence first, else highest-view f+1 speculative evidence."""

    commits = [item.commit_certificate for item in evidence if item.commit_certificate is not None]
    commit_values = {(item.block_digest, item.parent_digest) for item in commits}
    if len(commit_values) > 1:
        return None, None, "conflicting_commit_certificates"
    if commits:
        selected_commit = min(
            (item for item in commits if item.view == max(candidate.view for candidate in commits)),
            key=lambda item: item.deterministic_hash(),
        )
        return selected_commit, None, None

    grouped: dict[tuple[int, str, str], set[int]] = {}
    creation_times: dict[tuple[int, str, str], float] = {}
    for item in evidence:
        speculative = item.highest_speculative_evidence
        if speculative is None:
            continue
        key = (speculative.originating_view, speculative.block_digest, speculative.parent_digest)
        grouped.setdefault(key, set()).update(speculative.signer_ids)
        creation_times[key] = min(creation_times.get(key, speculative.creation_time), speculative.creation_time)
    qualified = {key: signers for key, signers in grouped.items() if len(signers) >= speculative_threshold}
    if not qualified:
        return None, None, None
    highest_view = max(key[0] for key in qualified)
    strongest = [(key, signers) for key, signers in qualified.items() if key[0] == highest_view]
    values = {(key[1], key[2]) for key, _ in strongest}
    if len(values) > 1:
        return None, None, "conflicting_highest_view_speculative_evidence"
    key, signers = strongest[0]
    selected_speculative = ZyzzyvaSpeculativeEvidence(
        evidence[0].epoch_id,
        evidence[0].height,
        key[0],
        key[1],
        key[2],
        tuple(signers),
        speculative_threshold,
        creation_times[key],
    )
    return None, selected_speculative, None


@dataclass(frozen=True, slots=True)
class ZyzzyvaNewViewCertificate(CanonicalSerializable):
    QUORUM_POLICY_VERSION = "liu_zyzzyva_view_change_quorum_v1"
    SAFE_VALUE_POLICY_VERSION = "liu_zyzzyva_safe_value_selection_v1"

    epoch_id: int
    height: int
    target_view: int
    new_primary_id: int
    evidence: tuple[ZyzzyvaViewChangeEvidence, ...]
    threshold: int
    speculative_threshold: int
    safe_block_digest: str | None
    safe_parent_digest: str | None
    carried_safety_evidence_hash: str | None
    selected_commit_certificate: ZyzzyvaCommitCertificate | None
    selected_speculative_evidence: ZyzzyvaSpeculativeEvidence | None
    creation_time: float

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.evidence, key=lambda item: item.sender_id))
        signer_ids = tuple(item.sender_id for item in ordered)
        if (
            self.target_view <= 0
            or self.threshold <= 0
            or self.speculative_threshold <= 0
            or len(ordered) != self.threshold
            or len(signer_ids) != len(set(signer_ids))
        ):
            raise ValueError("NEW_VIEW certificate requires exactly threshold distinct evidence signers")
        commit, speculative, error = select_zyzzyva_safe_value(ordered, self.speculative_threshold)
        if error is not None:
            raise ValueError(error)
        if commit != self.selected_commit_certificate or speculative != self.selected_speculative_evidence:
            raise ValueError("NEW_VIEW selected evidence does not follow safe-value policy")
        selected = commit if commit is not None else speculative
        expected_digest = None if selected is None else selected.block_digest
        expected_parent = None if selected is None else selected.parent_digest
        expected_hash = None if selected is None else selected.deterministic_hash()
        if (
            self.safe_block_digest != expected_digest
            or self.safe_parent_digest != expected_parent
            or self.carried_safety_evidence_hash != expected_hash
        ):
            raise ValueError("NEW_VIEW safe identity/evidence hash mismatch")
        object.__setattr__(self, "evidence", ordered)

    @property
    def signer_ids(self) -> tuple[int, ...]:
        return tuple(item.sender_id for item in self.evidence)

    def validate(
        self,
        replica_ids: tuple[int, ...],
        expected_primary_id: int,
        threshold: int,
        speculative_threshold: int,
        recovery_threshold: int,
    ) -> bool:
        if (
            self.new_primary_id != expected_primary_id
            or self.threshold != threshold
            or self.speculative_threshold != speculative_threshold
            or len(self.evidence) != threshold
            or not set(self.signer_ids).issubset(replica_ids)
        ):
            return False
        if any(
            item.epoch_id != self.epoch_id
            or item.height != self.height
            or item.target_view != self.target_view
            or not item.validate(replica_ids, recovery_threshold)
            for item in self.evidence
        ):
            return False
        commit, speculative, error = select_zyzzyva_safe_value(self.evidence, speculative_threshold)
        selected = commit if commit is not None else speculative
        return (
            error is None
            and commit == self.selected_commit_certificate
            and speculative == self.selected_speculative_evidence
            and self.safe_block_digest == (None if selected is None else selected.block_digest)
            and self.safe_parent_digest == (None if selected is None else selected.parent_digest)
            and self.carried_safety_evidence_hash == (None if selected is None else selected.deterministic_hash())
        )

    def to_dict(self) -> dict:
        return {
            "carried_safety_evidence_hash": self.carried_safety_evidence_hash,
            "creation_time": self.creation_time,
            "epoch_id": self.epoch_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "height": self.height,
            "new_primary_id": self.new_primary_id,
            "policy_version": self.SAFE_VALUE_POLICY_VERSION,
            "quorum_policy": self.QUORUM_POLICY_VERSION,
            "safe_block_digest": self.safe_block_digest,
            "safe_parent_digest": self.safe_parent_digest,
            "selected_commit_certificate": (
                None if self.selected_commit_certificate is None else self.selected_commit_certificate.to_dict()
            ),
            "selected_speculative_evidence": (
                None if self.selected_speculative_evidence is None else self.selected_speculative_evidence.to_dict()
            ),
            "speculative_threshold": self.speculative_threshold,
            "target_view": self.target_view,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class LiuQuorumReplyCertificate(CanonicalSerializable):
    block_identity: LiuBlockIdentity
    client_id: int
    signer_ids: tuple[int, ...]
    required_replies: int
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer_ids", _normalized_quorum_signers(self.signer_ids, self.required_replies))

    def validate(self, client_id: int, replica_ids: tuple[int, ...], identity: LiuBlockIdentity) -> bool:
        return (
            self.client_id == client_id
            and self.block_identity == identity
            and self.required_replies == len(replica_ids)
            and self.signer_ids == tuple(sorted(replica_ids))
        )

    def to_dict(self) -> dict:
        return {
            "block_identity": self.block_identity.to_dict(),
            "client_id": self.client_id,
            "created_at": self.created_at,
            "required_replies": self.required_replies,
            "signer_ids": list(self.signer_ids),
        }
