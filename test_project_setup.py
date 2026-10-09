from pathlib import Path

# This file lives at the project root, so parents[0] IS the project root.
ROOT = Path(__file__).resolve().parent


def test_data_directory_exists():
    assert (ROOT / "data").is_dir()


def test_data_directory_has_txt_files():
    assert list((ROOT / "data").glob("*.txt"))


def test_requirements_file_exists():
    assert (ROOT / "requirements.txt").is_file()


def test_templates_directory_has_index_html():
    assert (ROOT / "templates" / "index.html").is_file()


def test_gitignore_exists_and_excludes_env():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "env/" in gitignore
    assert "env (1)/" in gitignore


def test_tests_directory_has_init():
    assert (ROOT / "tests" / "__init__.py").is_file()
