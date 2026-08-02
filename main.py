"""Run lessons from here, or call them directly:

    uv run lessons/01_basic_chat.py
    uv run lessons/02_prompt_templates.py
"""

from pathlib import Path
import runpy
import sys


LESSONS = {
    "1": "01_basic_chat.py",
    "2": "02_prompt_templates.py",
}


def main() -> None:
    lessons_dir = Path(__file__).parent / "lessons"

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("LangChain lessons:")
        for key, name in LESSONS.items():
            print(f"  {key}. {name}")
        choice = input("\nRun lesson number: ").strip()

    filename = LESSONS.get(choice)
    if not filename:
        print(f"Unknown lesson: {choice}")
        sys.exit(1)

    runpy.run_path(str(lessons_dir / filename), run_name="__main__")


if __name__ == "__main__":
    main()
