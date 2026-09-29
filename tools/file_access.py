from pathlib import Path


def _clean_file_content(content):
    """
    Remove Markdown code fences if the LLM accidentally wraps
    file content in ```...```.

    Example:

        ```cpp
        int main() {}
        ```

    becomes:

        int main() {}
    """

    content = str(content).strip()

    if content.startswith("```"):
        lines = content.splitlines()

        # Remove opening fence
        if lines:
            lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines)

    return content


def list_files(path="."):
    """Return a visual tree of the directory structure."""

    def build_tree(dir_path, prefix=""):
        lines = []

        items = [
            p for p in Path(dir_path).iterdir()
            if p.name not in [".git", "__pycache__", ".venv"]
        ]

        for i, item in enumerate(
            sorted(
                items,
                key=lambda x: (x.is_file(), x.name.lower())
            )
        ):
            is_last = i == len(items) - 1

            connector = "└── " if is_last else "├── "

            lines.append(
                f"{prefix}{connector}{item.name}"
            )

            if item.is_dir():
                sub_prefix = (
                    "    " if is_last else "│   "
                )

                lines.extend(
                    build_tree(
                        item,
                        prefix=sub_prefix
                    )
                )

        return lines

    return "\n".join(build_tree(path))


def read_file(path):
    """Read and return the contents of a file."""

    file_path = Path(path)

    if not file_path.exists():
        return f"{path} does not exist."

    if not file_path.is_file():
        return f"{path} is not a file."

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


def write_file(path, content):
    """Create a new file or overwrite an existing file."""

    file_path = Path(path)

    # Create parent directories automatically.
    if file_path.parent != Path("."):
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    content = _clean_file_content(content)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(content)

    return f"Successfully wrote to {file_path}"


def append_file(path, content):
    """Append content to an existing file."""

    file_path = Path(path)

    if not file_path.exists():
        return f"{path} does not exist."

    content = _clean_file_content(content)

    with open(
        file_path,
        "a",
        encoding="utf-8"
    ) as file:
        file.write("\n" + content)

    return f"Successfully appended to {file_path}"


def dlt_file(path):
    """Delete a file."""

    file_path = Path(path)

    if not file_path.exists():
        return f"{path} does not exist."

    if not file_path.is_file():
        return f"{path} is not a file."

    file_path.unlink()

    return f"Deleted {path} successfully."


# ============================================================
# TOOL DESCRIPTIONS
# ============================================================

files_list = {
    "name": "list_files",
    "description": (
        "List files and directories inside a specified path. "
        "If no path is provided, list the current directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory path to list. "
                    "Defaults to current directory."
                )
            }
        }
    }
}


file_read = {
    "name": "read_file",
    "description": (
        "Read and return the contents of a file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to read."
            }
        },
        "required": ["path"]
    }
}


file_write = {
    "name": "write_file",
    "description": (
        "Create a new file or overwrite an existing file. "
        "Use this when the user explicitly asks to create, "
        "write, or overwrite a file. "
        "The content must be the actual file content, "
        "without Markdown code fences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to write."
            },
            "content": {
                "type": "string",
                "description": "Raw content to write into the file."
            }
        },
        "required": ["path", "content"]
    }
}


file_append = {
    "name": "append_file",
    "description": (
        "Append content to the end of an existing file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to modify."
            },
            "content": {
                "type": "string",
                "description": "Raw content to append."
            }
        },
        "required": ["path", "content"]
    }
}


file_delete = {
    "name": "dlt_file",
    "description": (
        "Delete a file from the filesystem."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to delete."
            }
        },
        "required": ["path"]
    }
}