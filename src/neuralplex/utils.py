from neuralplex.network import Network


def get_edge_data(network: Network):
    """Actual directed edges with signed connection weights (no null targets)."""
    return [
        {"source": source.name, "target": receiver.name, "weight": weight}
        for layer in network.layers
        for receiver in layer.neurons
        for source, weight in receiver.weights.items()
    ]
