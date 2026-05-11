import argparse
import json

from .data_loader import ProjectDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Przygotowanie danych dla wybranego tasku.")
    parser.add_argument("--task", required=True, help="Np. classification_divorce lub regression_years")
    parser.add_argument("--test-size", type=float, default=0.20, help="Rozmiar zbioru testowego, np. 0.2")
    parser.add_argument("--random-state", type=int, default=42, help="Seed")
    parser.add_argument("--no-scale", action="store_true", help="Wyłącz skalowanie zmiennych liczbowych")
    args = parser.parse_args()

    loader = ProjectDataLoader()
    prepared = loader.prepare(
        task_name=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        scale_numeric=not args.no_scale,
    )
    paths = loader.save_prepared_dataset(prepared)

    print(json.dumps({
        "saved": {k: str(v) for k, v in paths.items()},
        "metadata": prepared.metadata,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
