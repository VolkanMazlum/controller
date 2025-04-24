#### AI GENERATED!! USE AT YOUR OWN RISK!!

import os
import sys


def read_and_sort_lines(filepath):
    """
    Reads a file, strips leading/trailing whitespace from each line,
    filters out empty lines, and returns a sorted list of lines.

    Args:
        filepath (str): The path to the file.

    Returns:
        list: A sorted list of non-empty lines from the file,
              or None if the file cannot be read.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, "r") as f:
            # Read lines, strip whitespace, and ignore blank lines
            lines = [line.strip() for line in f if line.strip()]
        lines.sort()  # Sort the lines alphabetically
        return lines
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None


def compare_files_ignore_order(file1_path, file2_path):
    """
    Compares the content of two files, ignoring the order of lines.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.

    Returns:
        bool: True if the files contain the same lines (ignoring order),
              False otherwise.
    """
    print(
        f"Comparing '{os.path.basename(file1_path)}' and '{os.path.basename(file2_path)}'..."
    )

    lines1 = read_and_sort_lines(file1_path)
    if lines1 is None:
        return False  # Error occurred reading file 1

    lines2 = read_and_sort_lines(file2_path)
    if lines2 is None:
        return False  # Error occurred reading file 2

    if len(lines1) != len(lines2):
        print(f"Files have different number of non-empty lines:")
        print(f"  {os.path.basename(file1_path)}: {len(lines1)} lines")
        print(f"  {os.path.basename(file2_path)}: {len(lines2)} lines")
        print("\n\nWRONG\t\tResult: Files are DIFFERENT.")
        return False

    # Compare the sorted lists of lines
    if lines1 == lines2:
        print(f"\n\nCORRECT\t\tResult: Files contain the SAME data (ignoring order).")
        return True
    else:
        print("\n\nWRONG\t\tResult: Files contain DIFFERENT data.")
        # Optional: Find and print the first differing lines for debugging
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                print(f"  First difference found at sorted line index {i}:")
                print(f"    {os.path.basename(file1_path)}: '{l1}'")
                print(f"    {os.path.basename(file2_path)}: '{l2}'")
                break
        return False


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python verify_results_file.py <file1_path> <file2_path>")
        sys.exit(1)  # Exit with a non-zero status code indicates error

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Perform the comparison
    are_same = compare_files_ignore_order(file1, file2)

    # Exit with status code 0 if same, 1 if different or error
    sys.exit(0 if are_same else 1)
