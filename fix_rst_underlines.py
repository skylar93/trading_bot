import os
import re

def fix_underlines(content):
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is an underline
            if re.match(r'^[=\-\^\"\'~]+$', next_line):
                # Get the character used for underlining
                char = next_line[0]
                # Create new underline with correct length
                new_underline = char * len(line.rstrip())
                fixed_lines.append(line)
                fixed_lines.append(new_underline)
                i += 2
                continue
        fixed_lines.append(line)
        i += 1
    return '\n'.join(fixed_lines)

def process_rst_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.rst'):
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                fixed_content = fix_underlines(content)
                if content != fixed_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    print(f"Fixed underlines in {filepath}")

if __name__ == '__main__':
    process_rst_files('source') 