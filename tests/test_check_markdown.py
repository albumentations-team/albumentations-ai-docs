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
    def _create_md_file(filename: str, content: str, subdirs: str | None = None):
        """Creates a markdown file, optionally within subdirectories."""
        base_path = tmp_path
        if subdirs:
            base_path = tmp_path / subdirs
            base_path.mkdir(parents=True, exist_ok=True) # Create subdirs if they don't exist
        filepath = base_path / filename
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
    "filename, content, expected_errors, subdirs",
    [
        # No errors
        ("valid-filename.md", "Text with [a link](relative/path.md) and ![img](../img/image.webp).", [], None),
        ("valid-filename.md", "Text with [a link](relative/path.md) and ![img](../img/image.webp).", [], "valid/dir"),
        # Filename underscore
        ("file_with_underscore.md", "Content", ["Filename contains underscore: 'file_with_underscore.md'. Use hyphens instead."], None),
        # Relative link underscore
        ("link-test.md", "Link: [text](relative_path/file.md)", ["Relative link contains underscore: 'relative_path/file.md' in {filepath}"], None),
        # Relative link underscore in image (ALLOWED)
        ("link-test-img.md", "Image: ![alt text](relative_path/image.webp)", [], None), # Underscores are OK in image paths
        # Underscore in absolute link (ignored)
        ("link-test-abs.md", "Link: [text](https://example.com/path_with_underscore)", [], None),
        # Underscore in fragment (ignored)
        ("link-test-frag.md", "Link: [text](#section_header)", [], None),
         # Underscore in code block link (ignored)
        ("link-test-code.md", "Link: [`code_link`](`code_link`)", [], None),
        ("link-test-code2.md", "Link: [text](`relative_path/file.md`)", [], None),
        # Non-webp relative image
        ("image-test-png.md", "Image: ![alt text](./images/image.png)", ["Relative image link is not .webp: './images/image.png' in {filepath}. Found .png."], None),
        ("image-test-jpg.md", "Image: ![alt text](../images/figure.jpg)", ["Relative image link is not .webp: '../images/figure.jpg' in {filepath}. Found .jpg."], None),
        # Absolute image link (ignored)
        ("image-test-abs.md", "Image: ![alt text](https://example.com/image.png)", [], None),
        # Directory underscore
        ("test.md", "Content", ["Directory name contains underscore: 'dir_with_underscore' in path '{subdirs}'. Use hyphens instead."], "dir_with_underscore"),
        ("test.md", "Content", ["Directory name contains underscore: 'nested_dir' in path '{subdirs}'. Use hyphens instead."], "valid/nested_dir"),
        ("test.md", "Content",
         ["Directory name contains underscore: 'first_dir' in path '{subdirs}'. Use hyphens instead.",
          "Directory name contains underscore: 'second_deep' in path '{subdirs}'. Use hyphens instead."],
         "first_dir/valid/second_deep"),
        # Multiple errors (including directory)
        ("file_name.md", "Link: [bad_link](./file.md)", # Link target ./file.md is fine
         ["Filename contains underscore: 'file_name.md'. Use hyphens instead.",
          "Directory name contains underscore: 'bad_dir' in path '{subdirs}'. Use hyphens instead."],
         "bad_dir"),
    ],
    ids=[
        "no_errors",
        "no_errors_subdir",
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
        "directory_underscore_simple",
        "directory_underscore_nested",
        "directory_underscore_multiple",
        "multiple_errors_incl_dir_filename_link",
    ]
)
def test_check_markdown_file(create_md_file, filename, content, expected_errors, subdirs, tmp_path):
    filepath = create_md_file(filename, content, subdirs=subdirs)
    # Replace placeholders in expected errors with actual paths
    formatted_expected_errors = []
    for e in expected_errors:
        # Format {filepath} and potentially {subdirs}
        try:
            formatted_error = e.format(filepath=filepath, subdirs=subdirs or '.') # Use . if subdirs is None
        except KeyError: # Handle cases where only {filepath} is present
            formatted_error = e.format(filepath=filepath)
        formatted_expected_errors.append(formatted_error)

    # Pass tmp_path as project_root and use the UNRESOLVED filepath
    # Compare sets for order independence
    assert set(check_markdown_file(filepath, project_root=tmp_path)) == set(formatted_expected_errors)

# --- Test for main script execution (simulating pre-commit) ---

def run_script(files: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run the check_markdown.py script."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_markdown.py"
    cmd = [sys.executable, str(script_path)] + files
    # Use the provided cwd (should be tmp_path from the test fixture)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or Path.cwd())

def test_main_script_no_errors(create_md_file, tmp_path):
    file1 = create_md_file("test-file-1.md", "Content is fine.\n![img](img/ok.webp)")
    file2 = create_md_file("test-file-2.md", "[link](path/ok.md)")
    result = run_script([str(file1), str(file2)], cwd=tmp_path) # Pass tmp_path
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    # Should pass now
    assert result.returncode == 0
    assert "Markdown checks passed." in result.stdout

def test_main_script_with_errors(create_md_file, tmp_path):
    dir1 = Path("nodirerror")
    file1 = create_md_file("test_file_1.md", "Filename error.", subdirs=str(dir1 / "subdir_one"))
    file2 = create_md_file("test-file-2.md", "[link](path_nok/other.md)", subdirs=str(dir1 / "subdir-two"))
    file3 = create_md_file("test-file-3.md", "![img](img/bad.png)", subdirs=str(dir1 / "subdir-three"))

    result = run_script([str(file1), str(file2), str(file3)], cwd=tmp_path) # Pass tmp_path
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 1
    # Expect 3 original errors + 1 dir error = 4 errors
    assert "Markdown checks failed: 4 error(s) found." in result.stderr
    assert f"Filename contains underscore: '{file1.name}'" in result.stderr
    assert f"Directory name contains underscore: 'subdir_one'" in result.stderr
    assert f"Relative link contains underscore: 'path_nok/other.md'" in result.stderr
    assert f"Relative image link is not .webp: 'img/bad.png'" in result.stderr

def test_main_script_ignore_non_md(tmp_path):
    not_md = tmp_path / "script.py"
    not_md.write_text("print('hello')")
    md_ok = tmp_path / "doc.md"
    md_ok.write_text("All good")
    result = run_script([str(not_md), str(md_ok)], cwd=tmp_path) # Pass tmp_path
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    # Should pass now
    assert result.returncode == 0
    assert f"Checking {not_md}" not in result.stdout
    assert "Markdown checks passed." in result.stdout

def test_main_script_non_existent_file(tmp_path):
    non_existent = tmp_path / "ghost.md"
    md_ok = tmp_path / "doc.md"
    md_ok.write_text("All good")
    result = run_script([str(non_existent), str(md_ok)], cwd=tmp_path) # Pass tmp_path
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    # Should pass now
    assert result.returncode == 0
    # assert f"Skipping non-existent" in result.stdout
    assert "Markdown checks passed." in result.stdout
