import re

filepath = "docs/4-advanced-guides/test-time-augmentation.md"
with open(filepath, "r") as f:
    text = f.read()

transforms = [
    "HorizontalFlip", "VerticalFlip", "ColorJitter", "RandomGamma",
    "GaussNoise", "RandomBrightnessContrast", "SmallestMaxSize",
    "CenterCrop", "Normalize", "Crop", "Transpose", "Affine", "D4"
]

lines = text.split('\n')
new_lines = []
in_code = False

for line in lines:
    if line.startswith('```'):
        in_code = not in_code
        new_lines.append(line)
        continue

    if not in_code:
        modified_line = line
        for t in transforms:
            # We split by exact word matching the transform name
            pattern = re.compile(r'\b' + t + r'\b')
            parts = pattern.split(modified_line)

            # The parts list will alternate between text before the match, and text after.
            # actually re.split with capture f"({pattern})" keeps the matched word
            # Let's use re.finditer and build strings
            matches = list(pattern.finditer(modified_line))
            if not matches:
                continue

            new_line_str = ""
            last_end = 0
            for m in matches:
                start = m.start()
                end = m.end()

                # Context before the transform
                before = modified_line[:start]
                after = modified_line[end:]

                # Check if it looks like it's already a markdown link
                if (before.endswith('[') or before.endswith('[`')) and after.startswith(']'):
                    new_line_str += modified_line[last_end:end]
                elif 'explore.albumentations.ai/transform' in before:
                    new_line_str += modified_line[last_end:end]
                else:
                    new_line_str += modified_line[last_end:start] + f"[{t}](https://explore.albumentations.ai/transform/{t})"

                last_end = end

            new_line_str += modified_line[last_end:]
            modified_line = new_line_str

        new_lines.append(modified_line)
    else:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.write('\n'.join(new_lines))
print(f"Updated {filepath}")
