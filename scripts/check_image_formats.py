# scripts/check_image_formats.py
"""Check for forbidden image formats (PNG, JPG, JPEG) in the img/ directory."""

import re
import sys

IMG_DIR = "img/"


def main() -> int:
    """Check for forbidden image formats in the img directory."""
    forbidden_files_found = [
        filename
        for filename in sys.argv[1:]
        if filename.startswith(IMG_DIR) and re.search(r"\.(png|jpe?g)$", filename, re.IGNORECASE)
    ]

    if forbidden_files_found:
        sys.stderr.write(
            f"Error: Found forbidden image formats in '{IMG_DIR}'. Please use .webp instead.\n",
        )
        sys.stderr.write("Forbidden files:\n")
        for f in forbidden_files_found:
            sys.stderr.write(f"- {f}\n")
        return 1  # Indicate failure
    return 0  # Indicate success


if __name__ == "__main__":
    sys.exit(main())
