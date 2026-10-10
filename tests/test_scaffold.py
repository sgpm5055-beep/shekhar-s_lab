from pathlib import Path

import eligibility_agent


def test_package_importable_and_typed() -> None:
    package_dir = Path(eligibility_agent.__file__).parent
    assert (package_dir / "py.typed").is_file()
