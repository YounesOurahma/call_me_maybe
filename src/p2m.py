from pathlib import Path
import sys

# Directories and files to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
}

SCRIPT_NAME = Path(sys.argv[0]).name

IGNORE_FILES = {
    "project.md",
    SCRIPT_NAME,
}

ROOT = Path(".")
OUTPUT = Path("project.md")


def should_ignore(path: Path) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return True

    if path.name in IGNORE_FILES:
        return True

    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True

    return False


def write_tree(root: Path, out):
    out.write("# Project Structure\n\n")
    out.write("```text\n")

    for path in sorted(root.rglob("*")):
        if should_ignore(path):
            continue

        rel = path.relative_to(root)
        indent = "    " * (len(rel.parts) - 1)

        if path.is_dir():
            out.write(f"{indent}{rel.name}/\n")
        else:
            out.write(f"{indent}{rel.name}\n")

    out.write("```\n\n")


def language(ext: str) -> str:
    return {
        ".py": "python",
        ".md": "markdown",
        ".toml": "toml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".txt": "text",
        ".ini": "ini",
        ".cfg": "ini",
        ".sh": "bash",
        ".bat": "bat",
        ".html": "html",
        ".css": "css",
        ".js": "javascript",
        ".ts": "typescript",
    }.get(ext.lower(), "")


with OUTPUT.open("w", encoding="utf-8") as out:
    out.write("# Project Export\n\n")

    write_tree(ROOT, out)

    out.write("# Files\n\n")

    for file in sorted(ROOT.rglob("*")):
        if not file.is_file():
            continue

        if should_ignore(file):
            continue

        rel = file.relative_to(ROOT)

        out.write("---\n\n")
        out.write(f"## `{rel}`\n\n")

        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            out.write("*Unable to read file.*\n\n")
            continue

        out.write(f"```{language(file.suffix)}\n")
        out.write(text)

        if not text.endswith("\n"):
            out.write("\n")

        out.write("```\n\n")

print(f"Created {OUTPUT}")
