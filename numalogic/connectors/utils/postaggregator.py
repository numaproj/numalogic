from pydruid.utils.postaggregator import Field
from pydruid.utils.postaggregator import Postaggregator


class QuantilesDoublesSketchToQuantile(Postaggregator):
    """Class for building QuantilesDoublesSketchToQuantile post aggregator."""

    def __init__(self, field: Field, fraction: float, output_name=None):
        if output_name is None:
            name = "quantilesDoublesSketchToQuantile"
        else:
            name = output_name

        Postaggregator.__init__(self, None, None, name)
        self.field = field
        self.fraction = fraction
        self.post_aggregator = {
            "type": "quantilesDoublesSketchToQuantile",
            "name": name,
            "field": self.field.post_aggregator,
            "fraction": self.fraction,
        }
