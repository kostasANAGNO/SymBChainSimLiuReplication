import yaml
import sys

from Chain.NodeProfile import NodeProfileSet
from Chain.ValidatorSet import ValidatorSet


def read_yaml(path: str):
    """Reads a yaml file - assumes path is relevant to SBS_SRC"""
    with open(path, "rb") as f:
        data = yaml.safe_load(f)
    return data


class Parameters:
    """Contains all the parameters defined for the simulator and the simulation"""

    dynamic_sim = {}
    simulation = {}
    application = {}
    execution = {}
    data = {}
    consensus = {}
    network = {}
    reconfiguration = {}

    BigFoot = {}
    PBFT = {}
    Tendermint = {}
    LiuQuorum = {}
    LiuPBFT = {}
    LiuZyzzyva = {}

    behaviour = {}

    CPs = {}

    tx_factory = None

    global_configuration_chain = []

    # Static for the first validator-separation model. It deliberately remains
    # outside configuration blocks until dynamic membership is introduced.
    validator_set: ValidatorSet = None
    node_profile_set: NodeProfileSet = None
    _node_profile_config = None

    @staticmethod
    def reset_params():
        """resets all dictionaries storing the parameters of the system"""
        Parameters.dynamic_sim = {}
        Parameters.simulation = {}
        Parameters.application = {}
        Parameters.execution = {}
        Parameters.data = {}
        Parameters.consensus = {}
        Parameters.network = {}
        Parameters.behaviour = {}
        Parameters.CPs = {}

        Parameters.BigFoot = {}
        Parameters.PBFT = {}
        Parameters.Tendermint = {}
        Parameters.LiuQuorum = {}
        Parameters.LiuPBFT = {}
        Parameters.LiuZyzzyva = {}

        Parameters.reconfiguration = {}

        Parameters.global_configuration_chain = []
        Parameters.validator_set = None
        Parameters.node_profile_set = None
        Parameters._node_profile_config = None

    @staticmethod
    def load_params_from_config(config):
        """Parses config yaml file and initialises parameter dictionaries"""
        if "--config" in sys.argv:
            config = sys.argv[sys.argv.index("--config") + 1]

        params = read_yaml(f"../Configs/{config}")

        try:
            Parameters.dynamic_sim = params["dynamic_sim"]
        except KeyError:
            print("NO 'dynamic_sim' Parameters")

        try:
            Parameters.simulation = params["simulation"]
        except KeyError:
            print("NO 'simulation' Parameters")

        Parameters.simulation["events"] = {}  # cnt events of each type

        try:
            Parameters.behaviour = params["behaviour"]
        except KeyError:
            print("NO 'behaviour' Parameters")

        try:
            Parameters.network = params["network"]
        except KeyError:
            print("NO 'network' Parameters")

        try:
            Parameters.application = params["application"]
            Parameters.configure_validator_set()
            Parameters.calculate_fault_tolerance()
        except KeyError:
            print("NO 'application' Parameters")

        Parameters._node_profile_config = params.get("node_profiles")
        Parameters.configure_node_profiles()

        Parameters.application["txIDS"] = 0

        try:
            Parameters.execution = params["execution"]
        except KeyError:
            print("NO 'execution' Parameters")

        try:
            Parameters.data = params["data"]
        except KeyError:
            print("NO 'data' Parameters")

        Parameters.BigFoot = read_yaml(params["consensus"]["BigFoot"])
        Parameters.PBFT = read_yaml(params["consensus"]["PBFT"])
        Parameters.Tendermint = read_yaml(params["consensus"]["Tendermint"])
        Parameters.LiuQuorum = read_yaml(params["consensus"]["LiuQuorum"]) if params["consensus"].get("LiuQuorum") else {}
        Parameters.LiuPBFT = read_yaml(params["consensus"]["LiuPBFT"]) if params["consensus"].get("LiuPBFT") else {}
        Parameters.LiuZyzzyva = read_yaml(params["consensus"]["LiuZyzzyva"]) if params["consensus"].get("LiuZyzzyva") else {}

        try:
            Parameters.reconfiguration = params["reconfiguration"]
        except KeyError:
            print("NO 'reconfiguration' Parameters")

    @staticmethod
    def configure_validator_set():
        """Create deterministic static membership; omitted K means all nodes."""
        total_nodes = Parameters.application["Nn"]
        validator_count = Parameters.application.get("validator_count")
        Parameters.validator_set = ValidatorSet.first_nodes(total_nodes, validator_count)

    @staticmethod
    def configure_node_profiles():
        """Build one deterministic, ordered profile for every configured node."""
        Parameters.node_profile_set = NodeProfileSet.from_config(
            Parameters.application["Nn"],
            Parameters._node_profile_config,
        )

    @staticmethod
    def calculate_fault_tolerance():
        """Calculate the existing f and 2f+1 arithmetic over validators.

        This intentionally preserves ``int(K / 3)``. For K=21 it yields f=7
        and required_messages=15, rather than the classical
        floor((K-1)/3)=6 bound. Correcting that is a separate research change.
        """
        if Parameters.validator_set is None:
            Parameters.configure_validator_set()
        Parameters.application["f"] = int((1 / 3) * Parameters.validator_set.count)

        Parameters.application["required_messages"] = (2 * Parameters.application["f"]) + 1

    @staticmethod
    def parameters_to_string():
        """Returns a formatted string of all simulation parameters"""

        def dict_to_str(x, p_name_size=30):
            return "\n".join([f"{f'%{p_name_size}s' % key}: {value}" for key, value in x.items()])

        s = "-" * 20 + "DYNAMIC" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.dynamic_sim) + "\n"

        s += "-" * 20 + "SIMULATION" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.simulation) + "\n"

        s += "-" * 20 + "APPLICATION" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.application) + "\n"

        s += "-" * 20 + "EXECUTION" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.execution) + "\n"

        s += "-" * 20 + "DATA" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.data) + "\n"

        s += "-" * 20 + "NETWORK" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.network) + "\n"

        s += "-" * 20 + "BEHAVIOUR" + "-" * 20 + "\n"
        s += dict_to_str(Parameters.behaviour) + "\n"

        return s
