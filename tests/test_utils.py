from pathlib import Path

from src.utils import _matches_gitignore_pattern


def test_gitignore_glob_is_not_treated_as_regex():
    root = Path("/project")

    assert _matches_gitignore_pattern(
        root / "package.egg-info", root, "*.egg-info/"
    )
    assert not _matches_gitignore_pattern(root / "package.py", root, "*.egg-info/")


def test_gitignore_basename_pattern_matches_nested_directory():
    root = Path("/project")

    assert _matches_gitignore_pattern(root / "nested" / "outputs", root, "outputs/")


def test_gitignore_comments_and_blank_lines_do_not_match():
    root = Path("/project")

    assert not _matches_gitignore_pattern(root / "anything", root, "")
    assert not _matches_gitignore_pattern(root / "anything", root, "# comment")
