from pydruid.utils.postaggregator import Field
from pydruid.utils.postaggregator import Postaggregator


class QuantilesDoublesSketchToQuantile(Postaggregator):
    """Class for building QuantilesDoublesSketchToQuantile post aggregator."""

    def __init__(self, field: Field, fraction: float, output_name=None):
        name = output_name or "quantilesDoublesSketchToQuantile"

        super().__init__(None, None, name)
        self.field = field
        self.fraction = fraction
        self.post_aggregator = {
            "type": "quantilesDoublesSketchToQuantile",
            "name": name,
            "field": self.field.post_aggregator,
            "fraction": self.fraction,
        }
