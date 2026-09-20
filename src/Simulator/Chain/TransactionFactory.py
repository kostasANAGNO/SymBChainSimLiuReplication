from Parameters import Parameters

from Chain.Network import Network
from Utils.Instrumentation import InstrumentationCollector, TransactionCreationRecord

from collections import deque
import random

from typing import List, Deque, TYPE_CHECKING

if TYPE_CHECKING:
    from Chain.Node import Node
    from Chain.Block import Block

import logging

logger = logging.getLogger(__name__)


# Exact integer-byte accounting for block capacity (Liu Eq.(1) uses SI: 1 MB = 10^6 B,
# matching LiuReferenceParameters.megabytes_to_bytes = 1e6). Capacity/inclusion decisions
# MUST NOT depend on floating-point accumulation of MB-sized values (see the 999/1000
# investigation). Timing calculations (8*MB/Mbps) may remain floating point.
BYTES_PER_MB_EXACT: int = 1_000_000


def size_in_exact_bytes(size_mb: float) -> int:
    """Convert a MB-denominated size to exact integer bytes for capacity accounting.

    Uses SI 1 MB = 10^6 B (Liu convention), with a single rounding to remove the tiny
    representation error inherent in `0.0002 MB`-style floats. Values that are exact
    multiples of 1 B (e.g. paper grid `S_B in {0.2, 0.4, ...}` MB, tx sizes in whole
    bytes) round to the mathematically intended integer.
    """
    return round(float(size_mb) * BYTES_PER_MB_EXACT)


class Transaction:
    """
    Represents a single transaction in the blockchain system.

    Attributes:
        creator (int): ID of the node that created the transaction.
        id (int): Unique identifier for the transaction.
        timestamp (float): Time when the transaction was created.
        size (float): Size of the transaction in bytes.
        processed (bool): Whether the transaction has been processed/committed.
    """

    def __init__(
        self,
        creator: int,
        id: int,
        timestamp: float,
        size: float,
        original_creation_time: float = None,
    ) -> None:
        self.creator = creator
        self.id = id
        self._original_creation_time = timestamp if original_creation_time is None else original_creation_time
        self.available_at = timestamp
        self.size = size
        self.processed = False

    @property
    def original_creation_time(self) -> float:
        """Immutable timestamp at which the canonical transaction was created."""
        return self._original_creation_time

    @property
    def timestamp(self) -> float:
        """Backward-compatible alias for mempool availability time."""
        return self.available_at

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        self.available_at = value

    def __str__(self) -> str:
        return f"TX({self.timestamp},{self.size})"

    def __repr__(self) -> str:
        return self.__str__()


class TransactionFactory:
    """
    Handles the generation and execution of transactions and manages the memory pools of nodes.

    This class supports two types of transaction pool models:
    1) Local pool: Each node maintains its local transaction pool
    2) Global pool: A single transaction pool is maintained to improve simulation efficiency

    Class Attributes:
        nodes ([List['Node']]): The list of simulated nodes.
        produced_tx (int): Counter for transactions produced by the factory.
        global_mempool (Deque[Transaction]): Stores uncommitted transactions in global pool mode.
        depth_removed (int): Latest block depth for which committed transactions have been removed.
    """

    nodes: List["Node"] = []
    produced_tx: int = 0

    # GLOBAL TRANSACTION POOL MODEL
    global_mempool: Deque[Transaction] = deque([])  # global pool
    depth_removed: int = -1  # orchestrates removal of processed transactions

    @staticmethod
    def transaction_prop(tx: Transaction) -> None:
        """
        Models transaction propagation

        For local pool: Utilizes point-to-point delay between receiver and producer.
        For global pool: Utilizes an approximation based on creator's bandwidth.

        Args:
            tx (Transaction): The transaction to propagate.
        """
        match Parameters.application.get("transaction_model", "local"):
            case "local":
                for node in TransactionFactory.nodes:
                    if node.id == tx.creator:
                        continue
                    prop_delay = Network.calculate_message_propagation_delay(TransactionFactory.nodes[tx.creator], node, tx.size)
                    new_timestamp = tx.timestamp + prop_delay
                    tx = Transaction(tx.creator, tx.id, new_timestamp, tx.size, tx.original_creation_time)
                    node.pool.append(tx)
            case "global":
                prop_delay = Network.calculate_message_propagation_delay(
                    TransactionFactory.nodes[tx.creator],
                    TransactionFactory.nodes[tx.creator],
                    tx.size,
                )
                new_timestamp = tx.timestamp + prop_delay
                tx = Transaction(tx.creator, tx.id, new_timestamp, tx.size, tx.original_creation_time)
                TransactionFactory.global_mempool.append(tx)
            case _:
                logger.error(f"Unknown transaction model: '{Parameters.application['transaction_model']}'")
                raise (ValueError(f"Unknown transaction model: '{Parameters.application['transaction_model']}'"))

    @staticmethod
    def add_scenario_transactions(txion_list: List[tuple]) -> None:
        """
        Adds transactions from a list (provided by a scenario) to the simulation.

        Args:
            txion_list (List[tuple]): List of transactions in format (creator, id, timestamp, size).
        """
        logger.debug(f"Adding scenario transactions: {len(txion_list)} transactions")

        for creator, id, timestamp, size in txion_list:
            t = Transaction(creator, id, timestamp, size / 1e6)
            InstrumentationCollector.record_transaction_creation(
                TransactionCreationRecord(
                    transaction_id=t.id,
                    creator_node_id=t.creator,
                    original_creation_time=t.original_creation_time,
                    transaction_size=t.size,
                )
            )
            TransactionFactory.transaction_prop(t)

    @staticmethod
    def generate_interval_txions(start: float) -> None:
        """
        Generates transactions for the interval [start, start+TI_dur] based on the current configuration.

        Args:
            start (float): Start time for transaction generation.
        """
        logger.debug(f"Generating interval transactions starting at {start}")

        for second in range(round(start), round(start + Parameters.application["TI_dur"])):
            for _ in range(Parameters.application["Tn"]):
                if Parameters.simulation["stop_after_tx"] != -1 and TransactionFactory.produced_tx == Parameters.simulation["stop_after_tx"]:
                    logger.debug("Reached stop_after_tx limit, stopping transaction generation.")
                    return

                id = Parameters.application["txIDS"]
                Parameters.application["txIDS"] += 1

                timestamp = second

                size = random.expovariate(1 / Parameters.application["Tsize"])
                size += Parameters.application["base_transaction_size"]

                creator = random.choice(TransactionFactory.nodes)

                transaction = Transaction(creator.id, id, timestamp, size)
                InstrumentationCollector.record_transaction_creation(
                    TransactionCreationRecord(
                        transaction_id=transaction.id,
                        creator_node_id=transaction.creator,
                        original_creation_time=transaction.original_creation_time,
                        transaction_size=transaction.size,
                    )
                )
                TransactionFactory.transaction_prop(transaction)

                TransactionFactory.produced_tx += 1

    @staticmethod
    def execute_transactions(configuration, pool: Deque[Transaction], time: float) -> tuple[List[Transaction], float]:
        """
        Executes transactions by selecting them from the transaction pool.

        Args:
            configuration: The configuration settings for transaction execution.
            pool (Deque[Transaction]): The transaction pool to execute from.
            time (float): Current simulation time.

        Returns:
            tuple: (List of selected transactions, total size of transactions)
        """
        logger.debug(f"Executing transactions at time {time}")

        match Parameters.application.get("transaction_model", "local"):
            case "local":
                return TransactionFactory._get_transactions_from_pool(configuration, pool, time)
            case "global":
                return TransactionFactory._get_transactions_from_pool(configuration, TransactionFactory.global_mempool, time)
            case _:
                logger.error(f"Unknown transaction model: '{Parameters.application['transaction_model']}'")
                raise (ValueError(f"Unknown transaction model: '{Parameters.application['transaction_model']}'"))

    @staticmethod
    def _get_transactions_from_pool(configuration, pool: Deque[Transaction], time: float) -> tuple[List[Transaction], float]:
        """
        Retrieves transactions from the pool for inclusion in the next block.

        Args:
            configuration: Configuration containing block size limits.
            pool (Deque[Transaction]): Pool to get transactions from.
            time (float): Current simulation time.

        Returns:
            tuple: (List of selected transactions, total size or -1 if no valid transactions)
        """
        logger.debug(f"Getting transactions from pool at time {time}")

        transactions: List[Transaction] = []
        # Capacity accounting is exact integer bytes; the returned size stays in MB for
        # backwards-compatible callers (block.size metadata) but is derived from the
        # exact byte sum, so it never drifts due to float accumulation.
        capacity_bytes = size_in_exact_bytes(configuration.block_size)
        size_bytes: int = 0

        while pool:
            tx = pool[0]
            if tx.processed:
                pool.popleft()
                continue

            tx_bytes = size_in_exact_bytes(tx.size)
            if tx.available_at <= time and size_bytes + tx_bytes <= capacity_bytes:
                transactions.append(tx)
                size_bytes += tx_bytes
                pool.popleft()
            else:
                break

        if transactions:
            size_mb = size_bytes / BYTES_PER_MB_EXACT
            logger.debug(f"Selected {len(transactions)} transactions for block, total size: {size_mb}")
            return transactions, size_mb
        else:
            logger.debug("No valid transactions found for block.")
            return [], -1

    @staticmethod
    def mark_transactions_as_processed(block: "Block", pool: Deque[Transaction]) -> None:
        """
        Marks transactions in a block as processed in the appropriate pool.

        Args:
            block: The block containing transactions to mark.
            pool (Deque[Transaction]): The pool to mark transactions in.
        """
        logger.debug(f"Marking transactions as processed for block at depth {block.depth}")

        match Parameters.application["transaction_model"]:
            case "local":
                TransactionFactory._mark_pool(block.transactions, pool)
            case "global":
                # only one node needs to remove the transactions
                if TransactionFactory.depth_removed >= block.depth:
                    logger.debug(f"Transactions already removed for block depth {block.depth}")
                    return

                TransactionFactory._mark_pool(block.transactions, TransactionFactory.global_mempool)
                TransactionFactory.depth_removed = block.depth
                logger.debug(f"Transactions marked as processed for global pool at block depth {block.depth}")
            case _:
                logger.error(f"No such mempool model: {Parameters.application['transaction_model']}")
                raise ValueError(f"No such mempool model: {Parameters.application['transaction_model']}")

    @staticmethod
    def removed_processed(pool: Deque[Transaction]) -> Deque[Transaction]:
        """
        Returns a new pool without processed transactions.

        Args:
            pool (Deque[Transaction]): Pool to filter.
        Returns:
            Deque[Transaction]: New pool without processed transactions.
        """
        logger.debug("Removing processed transactions from pool")

        new_pool = deque([])
        while pool:
            tx = pool.popleft()
            if tx.processed:
                continue
            new_pool.append(tx)
        return new_pool

    @staticmethod
    def _mark_pool(txions: List[Transaction], pool: Deque[Transaction]) -> Deque[Transaction]:
        """
        Marks transactions in the pool as processed.

        Args:
            txions (List[Transaction]): Transactions to mark as processed.
            pool (Deque[Transaction]): Pool to mark transactions in.
        Returns:
            Deque[Transaction]: The updated pool.
        """
        logger.debug(f"Marking {len(txions)} transactions as processed in pool")

        unique_txions = set([tx.id for tx in txions])
        marked = 0

        for tx in pool:
            if tx.id in unique_txions:
                tx.processed = True
                marked += 1
            if marked == len(unique_txions):
                break

        return pool
