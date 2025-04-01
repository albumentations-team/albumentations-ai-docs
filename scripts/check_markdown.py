"""Markdown file checker script for pre-commit."""

import argparse
import re
import sys
from pathlib import Path

# Regex patterns
# Matches markdown links like [text](link), ensuring it's not preceded by !
MD_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)")
# Matches markdown image links like ![alt](link)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

# Common image extensions to check for (should be webp)
NON_WEBP_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


def is_relative(link: str) -> bool:
    """Check if a link is relative."""
    return not link.startswith(("http://", "https://", "#", "mailto:"))


# --- Helper Functions ---


def _check_filename(filename: str) -> list[str]:
    """Check filename for underscores."""
    if "_" in filename:
        return [f"Filename contains underscore: '{filename}'. Use hyphens instead."]
    return []


def _check_directories(filepath: Path, project_root: Path) -> list[str]:
    """Check directory path components for underscores relative to project root."""
    errors: list[str] = []
    try:
        # Ensure filepath is resolved relative to project_root if possible
        # Handle cases where filepath might already be absolute but inside project_root
        test_path = project_root / filepath if not filepath.is_absolute() else filepath

        relative_path = test_path.relative_to(project_root)

        # Iterate over parts of the directory path only
        for part in relative_path.parent.parts:
            if "_" in str(part):
                # Corrected error message format to match test expectations
                error_msg = (
                    f"Directory name contains underscore: '{part}' "
                    f"in path '{relative_path.parent}'. Use hyphens instead."
                )
                errors.append(error_msg)
    except ValueError:
        # This can happen if the resolved path is outside project_root, ignore.
        pass
    return errors


def _check_text_links(content: str, filepath: Path) -> list[str]:
    """Check relative text links for underscores."""
    errors = []
    text_links = MD_LINK_RE.findall(content)
    for _, link_target in text_links:
        if (
            is_relative(link_target)
            and not link_target.startswith("`")
            and not link_target.endswith("`")
            and "_" in link_target
        ):
            errors.append(f"Relative link contains underscore: '{link_target}' in {filepath}")
    return errors


def _fix_text_links(content: str, _filepath: Path | None = None) -> tuple[str, bool]:
    """Fix underscores in relative text links and return modified content and status."""
    modified_content = content
    content_changed = False
    text_links = MD_LINK_RE.findall(content)
    replacements_applied = set()  # Avoid duplicate replacements if link appears multiple times

    for link_text, link_target in text_links:
        original_md = f"[{link_text}]({link_target})"
        if (
            is_relative(link_target)
            and not link_target.startswith("`")
            and not link_target.endswith("`")
            and "_" in link_target
            and original_md not in replacements_applied  # Check if already processed
        ):
            fixed_target = link_target.replace("_", "-")
            fixed_md = f"[{link_text}]({fixed_target})"
            # Use regex substitution for safer replacement of the whole link
            # We need to escape potential regex metacharacters in link_text and link_target
            # Simpler approach: direct string replace if safe enough, but regex is more robust
            # For simplicity now, we'll stick to string replace, assuming links aren't too complex
            if original_md in modified_content:
                modified_content = modified_content.replace(original_md, fixed_md)
                replacements_applied.add(original_md)  # Mark as processed
                content_changed = True

    return modified_content, content_changed


def _check_image_links(content: str, filepath: Path) -> list[str]:
    """Check relative image links for non-webp extensions."""
    errors = []
    image_links = MD_IMAGE_RE.findall(content)
    for _alt_text, link_target in image_links:
        if is_relative(link_target):
            link_path = Path(link_target)
            if link_path.suffix.lower() in NON_WEBP_IMAGE_EXTENSIONS:
                errors.append(
                    f"Relative image link is not .webp: '{link_target}' in {filepath}. Found {link_path.suffix}.",
                )
    return errors


# --- Main Check Function ---


def check_markdown_file(filepath: Path, *, project_root: Path | None = None, fix: bool = False) -> list[str]:
    """Checks a single markdown file for violations, optionally fixing them."""
    errors = []
    filename = filepath.name

    # Check 1: Filename
    errors.extend(_check_filename(filename))

    # Check 1.5: Directories
    if project_root:
        errors.extend(_check_directories(filepath, project_root))

    # Read content
    try:
        original_content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        errors.append(f"Could not read file {filepath}: {e}")
        return errors  # Cannot proceed without content

    content_to_check = original_content
    modified_content = original_content
    content_changed_by_fix = False

    # Perform fixes if requested
    if fix:
        modified_content, content_changed_by_fix = _fix_text_links(original_content, filepath)
        if content_changed_by_fix:
            try:
                filepath.write_text(modified_content, encoding="utf-8")
                sys.stdout.write(f"Fixed underscores in relative links in: {filepath}\n")
                content_to_check = modified_content  # Check the fixed content
            except OSError as e:
                errors.append(f"Could not write fixes to file {filepath}: {e}")
                # If write fails, proceed to check original content for errors

    # Check 2: Text Links (on original or fixed content)
    # Only report errors if not fixing or if fixing didn't change the content
    if not fix or not content_changed_by_fix:
        errors.extend(_check_text_links(content_to_check, filepath))

    # Check 3: Image Links (on original or fixed content)
    errors.extend(_check_image_links(content_to_check, filepath))

    return errors


def main() -> None:
    """Runs the markdown checks on files provided via CLI arguments."""
    parser = argparse.ArgumentParser(description="Check Markdown file conventions.")
    parser.add_argument("files", nargs="*", help="Markdown files to check.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to automatically fix underscores in relative text links.",
    )
    args = parser.parse_args()

    markdown_files = [Path(f) for f in args.files if f.endswith(".md")]
    all_errors = []
    project_root = Path.cwd().resolve()  # Use resolved CWD

    for md_file in markdown_files:
        if md_file.exists() and md_file.is_file():
            # Pass project_root and fix flag
            file_errors = check_markdown_file(md_file, project_root=project_root, fix=args.fix)  # Pass fix by keyword
            all_errors.extend(file_errors)

    if all_errors:
        error_count = len(all_errors)
        sys.stderr.write(f"\n{error_count} Markdown convention violations found:\n")
        for error in all_errors:
            sys.stderr.write(f"- {error}\n")
        sys.exit(1)
    else:
        sys.stdout.write("\nMarkdown checks passed.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
