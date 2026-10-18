"""Скачиваем английскую документацию FastAPI в source_documents/."""

import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
DOCS_SUBPATH = "docs/en/docs"
CLONE_DIR = Path("_fastapi_clone")
OUTPUT_DIR = Path("source_documents")


def main():
    print("Клонируем репозиторий FastAPI (sparse checkout)...")
    clone_docs_only()
    print("Копируем markdown-файлы...")
    count = copy_docs()
    print("Подчищаем за собой...")
    shutil.rmtree(CLONE_DIR)
    print(f"Готово. {count} файлов в {OUTPUT_DIR}/")


def clone_docs_only():
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", REPO_URL, str(CLONE_DIR),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", DOCS_SUBPATH],
        cwd=CLONE_DIR,
        check=True,
    )


def copy_docs():
    src = CLONE_DIR / DOCS_SUBPATH
    OUTPUT_DIR.mkdir(exist_ok=True)
    count = 0
    for md_file in src.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue  # служебные страницы вроде _llm-test.md, это не документация
        relative = md_file.relative_to(src)
        dest = OUTPUT_DIR / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest)
        count += 1
    return count


if __name__ == "__main__":
    main()
