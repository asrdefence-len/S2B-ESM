import argparse

from scenarios import get_scenario, list_scenarios


DEFAULT_SCENARIO = "close_emitters"
DEFAULT_MHT = "gated"


def parse_args():
    parser = argparse.ArgumentParser(description="Run an S2B ESM simulation scenario.")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"Scenario name (default: {DEFAULT_SCENARIO})",
    )
    parser.add_argument(
        "--mht",
        choices=("gated", "reference"),
        default=DEFAULT_MHT,
        help=(
            "Association engine: gated (default, fast) or reference "
            "(slow probabilistic regression oracle)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    return parser.parse_args()


def print_scenarios():
    print("Available S2B ESM scenarios")
    print("---------------------------")
    for scenario in list_scenarios():
        print(f"{scenario.name:20s} {scenario.description}")


def select_scenario(args=None):
    if args is None:
        args = parse_args()

    if args.list:
        print_scenarios()
        return None

    return get_scenario(args.scenario)
