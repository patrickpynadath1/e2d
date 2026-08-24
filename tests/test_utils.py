from pathlib import Path

from src.utils import _flatten_and_copy, _matches_gitignore_pattern


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


def test_flatten_copy_accepts_gitignore_globs(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    ignored = source / "package.egg-info"
    ignored.mkdir()
    (ignored / "metadata.py").write_text("VALUE = 2\n", encoding="utf-8")

    destination = tmp_path / "flattened"
    _flatten_and_copy(source, destination, ["*.egg-info/"], source)

    assert (tmp_path / "flattened_model.py").read_text(encoding="utf-8") == (
        "VALUE = 1\n"
    )
    assert not (tmp_path / "flattened_package.egg-info_metadata.py").exists()
