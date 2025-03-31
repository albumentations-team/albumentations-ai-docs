import pytest
import subprocess
import sys
from pathlib import Path

# Add scripts directory to path to allow import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_markdown import is_relative, check_markdown_file, NON_WEBP_IMAGE_EXTENSIONS

# Fixture to create temporary markdown files
@pytest.fixture
def create_md_file(tmp_path):
    def _create_md_file(filename: str, content: str):
        filepath = tmp_path / filename
        filepath.write_text(content, encoding="utf-8")
        return filepath
    return _create_md_file

# --- Tests for is_relative ---

@pytest.mark.parametrize(
    "link, expected",
    [
        ("http://example.com", False),
        ("https://example.com/path", False),
        ("#section", False),
        ("mailto:test@example.com", False),
        ("relative/path.md", True),
        ("../relative/path.html", True),
        ("/absolute/but/not/url", True), # Treat server-absolute paths as relative for checks
        ("just_a_filename.md", True),
    ],
)
def test_is_relative(link, expected):
    assert is_relative(link) == expected

# --- Tests for check_markdown_file ---

@pytest.mark.parametrize(
    "filename, content, expected_errors",
    [
        # No errors
        ("valid-filename.md", "Text with [a link](relative/path.md) and ![img](../img/image.webp).", []),
        # Filename underscore
        ("file_with_underscore.md", "Content", ["Filename contains underscore: 'file_with_underscore.md'. Use hyphens instead."]),
        # Relative link underscore
        ("link-test.md", "Link: [text](relative_path/file.md)", ["Relative link contains underscore: 'relative_path/file.md' in {filepath}"]),
        # Relative link underscore in image (ALLOWED)
        ("link-test-img.md", "Image: ![alt text](relative_path/image.webp)", []), # Underscores are OK in image paths
        # Underscore in absolute link (ignored)
        ("link-test-abs.md", "Link: [text](https://example.com/path_with_underscore)", []),
        # Underscore in fragment (ignored)
        ("link-test-frag.md", "Link: [text](#section_header)", []),
         # Underscore in code block link (ignored)
        ("link-test-code.md", "Link: [`code_link`](`code_link`)", []),
        ("link-test-code2.md", "Link: [text](`relative_path/file.md`)", []),
        # Non-webp relative image
        ("image-test-png.md", "Image: ![alt text](./images/image.png)", ["Relative image link is not .webp: './images/image.png' in {filepath}. Found .png."]),
        ("image-test-jpg.md", "Image: ![alt text](../images/figure.jpg)", ["Relative image link is not .webp: '../images/figure.jpg' in {filepath}. Found .jpg."]),
        # Absolute image link (ignored)
        ("image-test-abs.md", "Image: ![alt text](https://example.com/image.png)", []),
        # Multiple errors
        ("multiple_errors.md", "Link: [text](relative_path/file.md)\nImage: ![alt](./img/pic.jpeg)", [
            "Filename contains underscore: 'multiple_errors.md'. Use hyphens instead.",
            "Relative link contains underscore: 'relative_path/file.md' in {filepath}",
            "Relative image link is not .webp: './img/pic.jpeg' in {filepath}. Found .jpeg."
        ]),
    ],
    ids=[
        "no_errors",
        "filename_underscore",
        "relative_link_underscore",
        "relative_image_link_underscore_allowed",
        "absolute_link_underscore_ignored",
        "fragment_underscore_ignored",
        "code_block_link_ignored",
        "code_block_link_target_ignored",
        "relative_image_png",
        "relative_image_jpg",
        "absolute_image_ignored",
        "multiple_errors",
    ]
)
def test_check_markdown_file(create_md_file, filename, content, expected_errors):
    filepath = create_md_file(filename, content)
    # Replace placeholder in expected errors with actual path
    formatted_expected_errors = [e.format(filepath=filepath) for e in expected_errors]
    assert check_markdown_file(filepath) == formatted_expected_errors

# --- Test for main script execution (simulating pre-commit) ---

def run_script(files: list[str]) -> subprocess.CompletedProcess:
    """Helper to run the check_markdown.py script."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_markdown.py"
    cmd = [sys.executable, str(script_path)] + files
    return subprocess.run(cmd, capture_output=True, text=True)

def test_main_script_no_errors(create_md_file):
    file1 = create_md_file("test-file-1.md", "Content is fine.\n![img](img/ok.webp)")
    file2 = create_md_file("test-file-2.md", "[link](path/ok.md)")
    result = run_script([str(file1), str(file2)])
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0
    assert "Markdown checks passed." in result.stdout

def test_main_script_with_errors(create_md_file):
    file1 = create_md_file("test_file_1.md", "Filename error.") # Filename error
    file2 = create_md_file("test-file-2.md", "[link](path_nok/other.md)") # Link error
    file3 = create_md_file("test-file-3.md", "![img](img/bad.png)") # Image error
    result = run_script([str(file1), str(file2), str(file3)])
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 1
    assert "Markdown checks failed:" in result.stdout
    assert f"Filename contains underscore: '{file1.name}'" in result.stdout
    assert f"Relative link contains underscore: 'path_nok/other.md'" in result.stdout
    assert f"Relative image link is not .webp: 'img/bad.png'" in result.stdout

def test_main_script_ignore_non_md(tmp_path):
    not_md = tmp_path / "script.py"
    not_md.write_text("print('hello')")
    md_ok = tmp_path / "doc.md"
    md_ok.write_text("All good")
    result = run_script([str(not_md), str(md_ok)])
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0
    assert f"Checking {md_ok}" in result.stdout
    assert f"Checking {not_md}" not in result.stdout # Should be skipped by main()
    assert "Markdown checks passed." in result.stdout

def test_main_script_non_existent_file(tmp_path):
    non_existent = tmp_path / "ghost.md"
    md_ok = tmp_path / "doc.md"
    md_ok.write_text("All good")
    result = run_script([str(non_existent), str(md_ok)])
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0 # Non-existent files are currently skipped gracefully
    assert f"Checking {md_ok}" in result.stdout
    # assert f"Skipping non-existent" in result.stdout # If we uncomment the print
    assert "Markdown checks passed." in result.stdout 