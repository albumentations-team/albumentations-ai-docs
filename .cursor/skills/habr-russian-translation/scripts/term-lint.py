#!/usr/bin/env python3
"""Lightweight linter for RU anti-anglicism rules in prose files.

Intentionally conservative:
- catches obvious hybrid verbs and weak transliterations,
- skips fenced code and inline code,
- helps with first-pass review,
- does not replace manual editorial judgment.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTS = {".md", ".txt", ".rst"}


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    replacement: str
    reason: str


RULES: tuple[Rule, ...] = (
    # --- Hybrid verbs (kill on sight) ---
    Rule(re.compile(r"\bзаапплаить\b", re.IGNORECASE), "применить", "verb anglicism"),
    Rule(re.compile(r"\bматчить\b", re.IGNORECASE), "соответствовать", "verb anglicism"),
    Rule(re.compile(r"\bхэндлить\b", re.IGNORECASE), "обрабатывать", "verb anglicism"),
    Rule(re.compile(r"\bскейлить\b", re.IGNORECASE), "масштабировать", "verb anglicism"),
    Rule(re.compile(r"\bимплементить\b", re.IGNORECASE), "реализовать", "verb anglicism"),
    Rule(re.compile(r"\bимплементировать\b", re.IGNORECASE), "реализовать", "heavy calque"),
    Rule(re.compile(r"\bконтрибьютить\b", re.IGNORECASE), "вносить вклад", "verb anglicism"),
    Rule(re.compile(r"\bсетапить\b", re.IGNORECASE), "настроить", "verb anglicism"),
    Rule(re.compile(r"\bимпактит\b", re.IGNORECASE), "влияет", "verb anglicism"),
    Rule(re.compile(r"\bпофиксить\b", re.IGNORECASE), "исправить", "verb anglicism"),
    Rule(re.compile(r"\bфиксануть\b", re.IGNORECASE), "исправить", "verb anglicism"),
    Rule(re.compile(r"\bзааблейтить\b", re.IGNORECASE), "провести абляцию", "verb anglicism"),
    Rule(re.compile(r"\bзарегуляризить\b", re.IGNORECASE), "добавить регуляризацию", "verb anglicism"),
    Rule(re.compile(r"\bзаинференсить\b", re.IGNORECASE), "прогнать инференс", "verb anglicism"),
    Rule(re.compile(r"\bотдебажить\b", re.IGNORECASE), "отладить", "verb anglicism"),
    Rule(re.compile(r"\bзадеплоить\b", re.IGNORECASE), "развернуть/выкатить", "verb anglicism"),
    Rule(re.compile(r"\bпротюнить\b", re.IGNORECASE), "настроить/подобрать", "verb anglicism"),
    Rule(re.compile(r"\bперетрейнить\b", re.IGNORECASE), "переобучить", "verb anglicism"),
    Rule(re.compile(r"\bпроаугментировать\b", re.IGNORECASE), "применить аугментацию", "verb anglicism"),
    Rule(re.compile(r"\bзалоггировать\b", re.IGNORECASE), "записать в лог", "verb anglicism"),
    Rule(re.compile(r"\bтюнить\b", re.IGNORECASE), "настраивать/подбирать", "verb anglicism"),
    Rule(re.compile(r"\bконвертить\b", re.IGNORECASE), "преобразовать", "verb anglicism"),
    Rule(re.compile(r"\bчекнуть\b", re.IGNORECASE), "проверить", "verb anglicism"),
    Rule(re.compile(r"\bоверсэмплить\b", re.IGNORECASE), "увеличить долю примеров", "verb anglicism"),
    Rule(re.compile(r"\bаугментить\b", re.IGNORECASE), "применить аугментацию", "verb anglicism"),
    # --- Bad transliterations ---
    Rule(re.compile(r"\bботтлнек\b", re.IGNORECASE), "узкое место", "bad transliteration"),
    Rule(re.compile(r"\bперформанс\b", re.IGNORECASE), "качество/метрики", "bad transliteration"),
    Rule(re.compile(r"\bлейбл\b", re.IGNORECASE), "метка", "bad transliteration"),
    Rule(re.compile(r"\bфича\b", re.IGNORECASE), "признак/возможность", "bad transliteration"),
    Rule(re.compile(r"\bробастность\b", re.IGNORECASE), "устойчивость", "bad transliteration"),
    Rule(re.compile(r"\bкапасити\b", re.IGNORECASE), "ёмкость", "bad transliteration"),
    Rule(re.compile(r"\bтрейд-офф\b", re.IGNORECASE), "компромисс/баланс", "bad transliteration"),
    Rule(re.compile(r"\bюзкейс\b", re.IGNORECASE), "сценарий использования", "bad transliteration"),
    Rule(re.compile(r"\bсетап\b", re.IGNORECASE), "настройка/конфигурация", "bad transliteration"),
    Rule(re.compile(r"\bоверхед\b", re.IGNORECASE), "накладные расходы", "bad transliteration"),
    Rule(re.compile(r"\bоверфиттинг\b", re.IGNORECASE), "переобучение", "bad transliteration"),
    Rule(re.compile(r"\bандерфиттинг\b", re.IGNORECASE), "недообучение", "bad transliteration"),
    Rule(re.compile(r"\bмагнитуда\b", re.IGNORECASE), "интенсивность/сила", "bad transliteration"),
    Rule(re.compile(r"\bаутпут\b", re.IGNORECASE), "выход/результат", "bad transliteration"),
    Rule(re.compile(r"\bинпут\b", re.IGNORECASE), "вход/входные данные", "bad transliteration"),
    Rule(re.compile(r"\bстейт\b", re.IGNORECASE), "состояние", "bad transliteration"),
    Rule(re.compile(r"\bрисёрч\b", re.IGNORECASE), "исследование", "bad transliteration"),
    Rule(re.compile(r"\bэвиденция\b", re.IGNORECASE), "доказательства/результаты", "bad transliteration"),
    Rule(re.compile(r"\bконфиденс\b", re.IGNORECASE), "уверенность", "bad transliteration"),
    Rule(re.compile(r"\bтейкауэй\b", re.IGNORECASE), "главный вывод", "bad transliteration"),
    Rule(re.compile(r"\bворкфлоу\b", re.IGNORECASE), "процесс/порядок работы", "bad transliteration"),
    # --- Bureaucratic filler ---
    Rule(re.compile(r"\bданный\b", re.IGNORECASE), "этот", "bureaucratic filler (данный -> этот)"),
    Rule(re.compile(r"\bданная\b", re.IGNORECASE), "эта", "bureaucratic filler (данная -> эта)"),
    Rule(re.compile(r"\bданное\b", re.IGNORECASE), "это", "bureaucratic filler (данное -> это)"),
    Rule(re.compile(r"\bосуществляется\b", re.IGNORECASE), "[rewrite with direct verb]", "bureaucratic verb"),
    Rule(re.compile(r"\bосуществлять\b", re.IGNORECASE), "[rewrite with direct verb]", "bureaucratic verb"),
    # --- False friends ---
    Rule(re.compile(r"\bколлапсирует\b", re.IGNORECASE), "ломается/проседает", "false friend (collapse)"),
    Rule(re.compile(r"\bколлапсировать\b", re.IGNORECASE), "ломаться/проседать", "false friend (collapse)"),
)

# Cyrillic + Latin mixed token, common marker of hybrid slang.
MIXED_SCRIPT_TOKEN = re.compile(r"\b[а-яё]+[a-z]+[а-яё]*\b|\b[a-z]+[а-яё]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    col_no: int
    token: str
    reason: str
    replacement: str


def iter_files(paths: Iterable[Path], extensions: set[str]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            if root.suffix.lower() in extensions:
                yield root
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                yield p


def lint_file(path: Path, check_mixed: bool) -> list[Finding]:
    findings: list[Finding] = []
    in_fence = False

    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if "`" in line:
            line = re.sub(r"`[^`]*`", "", line)

        # Skip lines that are table headers or table separators
        if stripped.startswith("|") and ("---" in stripped or stripped.endswith("|")):
            # Still check, but note that table content may have examples
            pass

        for rule in RULES:
            for match in rule.pattern.finditer(line):
                findings.append(
                    Finding(
                        path=path,
                        line_no=i,
                        col_no=match.start() + 1,
                        token=match.group(0),
                        reason=rule.reason,
                        replacement=rule.replacement,
                    )
                )

        if check_mixed:
            for match in MIXED_SCRIPT_TOKEN.finditer(line):
                token = match.group(0)
                findings.append(
                    Finding(
                        path=path,
                        line_no=i,
                        col_no=match.start() + 1,
                        token=token,
                        reason="mixed-script token",
                        replacement="review manually",
                    )
                )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint RU prose for anglicism issues.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    parser.add_argument(
        "--ext",
        nargs="*",
        default=sorted(DEFAULT_EXTS),
        help="File extensions to include (default: .md .txt .rst).",
    )
    parser.add_argument(
        "--no-mixed-script",
        action="store_true",
        help="Disable mixed Cyrillic/Latin token detection.",
    )
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.paths]
    exts = {e if e.startswith(".") else f".{e}" for e in args.ext}

    findings: list[Finding] = []
    for file_path in iter_files(roots, exts):
        findings.extend(lint_file(file_path, check_mixed=not args.no_mixed_script))

    if not findings:
        print("term-lint: OK")
        return 0

    for f in findings:
        print(
            f"{f.path}:{f.line_no}:{f.col_no}: {f.reason}: '{f.token}'"
            f" -> suggested '{f.replacement}'"
        )

    print(f"\nterm-lint: found {len(findings)} issue(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
