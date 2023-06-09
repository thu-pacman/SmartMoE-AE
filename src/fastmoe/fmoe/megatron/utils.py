r"""
Utility in Megatron
"""
from importlib.metadata import requires


def add_fmoe_args(parser):
    group = parser.add_argument_group(title="fastmoe")

    group.add_argument("--fmoefy", action="store_true")
    group.add_argument("--num-experts", type=int, default=None)
    group.add_argument("--top-k", type=int, default=2)
    group.add_argument("--balance-loss-weight", type=float, default=1)
    group.add_argument("--balance-strategy", type=str, default=None)
    group.add_argument("--hidden-hidden-size", type=int, default=None)
    group.add_argument("--expert-ep-size", type=int, default=None)
    group.add_argument("--expert-dp-size", type=int, default=None)
    group.add_argument("--dynamic-placement", action="store_true")
    group.add_argument("--dynamic-freq", type=int, default=10)
    group.add_argument("--gshard-cap", type=float, default=1.2)

    return parser
