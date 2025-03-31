import sys
import re
import os
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

def check_markdown_file(filepath: Path) -> list[str]:
    """Checks a single markdown file for violations."""
    errors = []
    filename = filepath.name

    # Check 1: Filename underscores
    if "_" in filename:
        errors.append(f"Filename contains underscore: '{filename}'. Use hyphens instead.")

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        errors.append(f"Could not read file {filepath}: {e}")
        return errors # Cannot check content if reading failed

    # Check 2: Underscores in relative non-image links
    text_links = MD_LINK_RE.findall(content)
    for _link_text, link_target in text_links: # Extract link_target from the tuple
        if is_relative(link_target):
            # Ignore links within code blocks
            if not link_target.startswith("`") and not link_target.endswith("`"):
                if "_" in link_target:
                    errors.append(f"Relative link contains underscore: '{link_target}' in {filepath}")

    # Check 3: Relative image link extensions must be .webp (underscores allowed in path)
    image_links = MD_IMAGE_RE.findall(content)
    for _alt_text, link_target in image_links: # _alt_text is unused
        if is_relative(link_target):
            link_path = Path(link_target)
            # Check extension
            if link_path.suffix.lower() in NON_WEBP_IMAGE_EXTENSIONS:
                errors.append(f"Relative image link is not .webp: '{link_target}' in {filepath}. Found {link_path.suffix}.")
            # Do not check for underscores in image links here

    return errors

def main():
    markdown_files = [Path(f) for f in sys.argv[1:] if f.endswith(".md")]
    all_errors = []

    for md_file in markdown_files:
        if md_file.exists() and md_file.is_file():
             print(f"Checking {md_file}...")
             file_errors = check_markdown_file(md_file)
             all_errors.extend(file_errors)
        # else:
        #     print(f"Skipping non-existent or non-file path: {md_file}")


    if all_errors:
        print("\nMarkdown checks failed:")
        for error in all_errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("\nMarkdown checks passed.")
        sys.exit(0)

if __name__ == "__main__":
    main() 