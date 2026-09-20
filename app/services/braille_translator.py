"""Liblouis Braille Translation Service.

Routes text into Liblouis bindings / in-memory UEB translation engine to produce
Grade 1 (uncontracted) and Grade 2 (contracted) Braille ASCII and Braille Unicode directly in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, cast

if TYPE_CHECKING:
    from app.services.document_parser import DocumentBlock
import re
from app.config import settings
from app.core.exceptions import TranslationError
from app.core.logging import logger
from app.models.schemas import BrailleTranslation

BRAILLE_ASCII_TO_UNICODE: dict = {
    " ": "\u2800", "!": "\u282e", '"': "\u2810", "#": "\u283c", "$": "\u282b", "%": "\u2829", "&": "\u282f", "'": "\u2804",
    "(": "\u2837", ")": "\u283e", "*": "\u2816", "+": "\u282c", ",": "\u2820", "-": "\u2824", ".": "\u2828", "/": "\u280c",
    "0": "\u2834", "1": "\u2802", "2": "\u2806", "3": "\u2812", "4": "\u2832", "5": "\u2822", "6": "\u2816", "7": "\u2836",
    "8": "\u2826", "9": "\u2814", ":": "\u2831", ";": "\u2830", "<": "\u2823", "=": "\u283f", ">": "\u281c", "?": "\u2839",
    "@": "\u2808", "A": "\u2801", "B": "\u2803", "C": "\u2809", "D": "\u2819", "E": "\u2811", "F": "\u280b", "G": "\u281b",
    "H": "\u2813", "I": "\u280a", "J": "\u281a", "K": "\u2805", "L": "\u2807", "M": "\u280d", "N": "\u281d", "O": "\u2815",
    "P": "\u280f", "Q": "\u281f", "R": "\u2817", "S": "\u280e", "T": "\u281e", "U": "\u2825", "V": "\u2827", "W": "\u283a",
    "X": "\u282d", "Y": "\u283d", "Z": "\u2835", "[": "\u282a", "\\": "\u2833", "]": "\u283b", "^": "\u2818", "_": "\u2838",
    "a": "\u2801", "b": "\u2803", "c": "\u2809", "d": "\u2819", "e": "\u2811", "f": "\u280b", "g": "\u281b", "h": "\u2813",
    "i": "\u280a", "j": "\u281a", "k": "\u2805", "l": "\u2807", "m": "\u280d", "n": "\u281d", "o": "\u2815", "p": "\u280f",
    "q": "\u281f", "r": "\u2817", "s": "\u280e", "t": "\u281e", "u": "\u2825", "v": "\u2827", "w": "\u283a", "x": "\u282d",
    "y": "\u283d", "z": "\u2835",
}

UEB_WHOLE_WORD_CONTRACTIONS: dict = {
    "and": "&", "for": "=", "of": "(", "the": "!", "with": ")", "that": "t", "this": "?", "which": ":", "but": "b",
    "can": "c", "do": "d", "every": "e", "from": "f", "go": "g", "have": "h", "just": "j", "knowledge": "k", "like": "l",
    "more": "m", "not": "n", "people": "p", "quite": "q", "rather": "r", "so": "s", "us": "u", "very": "v", "will": "w",
    "it": "x", "you": "y", "as": "z", "child": "*", "shall": "%", "out": "\\", "still": "/",
}

UEB_SUBSTRING_CONTRACTIONS: list = [
    ("and", "&"), ("for", "="), ("of", "("), ("the", "!"), ("with", ")"), ("ch", "*"), ("gh", "<"), ("sh", "%"), ("th", "?"),
    ("wh", ":"), ("ed", "$"), ("er", "]"), ("ou", "\\"), ("ow", "["), ("st", "/"), ("ar", ">"), ("ing", "+"), ("en", "5"),
    ("in", "9"), ("ea", "2"), ("bb", "2"), ("cc", "3"), ("dd", "4"), ("ff", "6"), ("gg", "7"),
]

class BrailleTranslator:
    def __init__(self) -> None:
        self._louis_module = self._try_import_louis()

    def _try_import_louis(self) -> Optional[object]:
        try:
            import louis
            return cast(Optional[object], louis)
        except (ImportError, OSError):
            return None

    def is_liblouis_available(self) -> Tuple[bool, Optional[str]]:
        if self._louis_module is not None:
            try:
                version = getattr(self._louis_module, "version", lambda: "native")()
                return True, str(version)
            except Exception:
                return True, "native"
        return False, None

    def ascii_to_unicode(self, braille_ascii: str) -> str:
        result = []
        for char in braille_ascii:
            if char in ("\n", "\r", "\t"):
                result.append(char)
            else:
                result.append(BRAILLE_ASCII_TO_UNICODE.get(char, "\u2800"))
        return "".join(result)

    def _translate_g1_fallback(self, text: str) -> str:
        result = []
        in_number_mode = False
        i = 0
        while i < len(text):
            char = text[i]
            if char in (" ", "\n", "\r", "\t"):
                in_number_mode = False
                result.append(char)
                i += 1
                continue
            if char.isdigit():
                if not in_number_mode:
                    result.append("#")
                    in_number_mode = True
                digit_map = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f", "7": "g", "8": "h", "9": "i", "0": "j"}
                result.append(digit_map[char])
                i += 1
                continue
            in_number_mode = False
            if char.isupper():
                word_match = re.match(r"^[A-Z]{2,}\b", text[i:])
                if word_match and (i == 0 or text[i - 1] in (" ", "\n", "\t")):
                    cap_word = word_match.group(0)
                    result.append(",,")
                    result.append(cap_word.lower())
                    i += len(cap_word)
                    continue
                else:
                    result.append(",")
                    result.append(char.lower())
                    i += 1
                    continue
            punct_map = {",": "1", ";": "2", ":": "3", ".": "4", "!": "6", "(": "7", ")": "7", "?": "8", '"': "8", "'": "'", "-": "-", "/": "/", "&": "&", "$": "$", "%": "%", "*": "*", "@": "@"}
            if char in punct_map:
                result.append(punct_map[char])
                i += 1
                continue
            if "a" <= char <= "z":
                result.append(char)
                i += 1
                continue
            result.append(char if ord(char) < 128 else " ")
            i += 1
        return "".join(result)

    def _translate_g2_fallback(self, text: str) -> str:
        lines = text.split("\n")
        translated_lines = []
        for line in lines:
            words = re.findall(r"\S+|\s+", line)
            translated_words = []
            for token in words:
                if token.isspace():
                    translated_words.append(token)
                    continue
                match = re.match(r"^([^\w]*)([\w'-]+)([^\w]*)$", token)
                if match:
                    prefix, core_word, suffix = match.groups()
                    lower_word = core_word.lower()
                    if lower_word in UEB_WHOLE_WORD_CONTRACTIONS:
                        contracted = UEB_WHOLE_WORD_CONTRACTIONS[lower_word]
                        if core_word.isupper() and len(core_word) > 1:
                            contracted_token = f",,{contracted}"
                        elif core_word[0].isupper():
                            contracted_token = f",{contracted}"
                        else:
                            contracted_token = contracted
                        trans_prefix = self._translate_g1_fallback(prefix)
                        trans_suffix = self._translate_g1_fallback(suffix)
                        translated_words.append(f"{trans_prefix}{contracted_token}{trans_suffix}")
                        continue
                g1_trans = self._translate_g1_fallback(token)
                for sub, braille_sub in UEB_SUBSTRING_CONTRACTIONS:
                    g1_trans = g1_trans.replace(sub, braille_sub)
                translated_words.append(g1_trans)
            translated_lines.append("".join(translated_words))
        return "\n".join(translated_lines)

    def translate(self, text: str, table_g1: Optional[str] = None, table_g2: Optional[str] = None) -> BrailleTranslation:
        if not text:
            return BrailleTranslation(grade_1_ascii="", grade_2_ascii="", grade_1_unicode="", grade_2_unicode="", table_g1_used=table_g1 or settings.LIBLOUIS_DEFAULT_G1_TABLE, table_g2_used=table_g2 or settings.LIBLOUIS_DEFAULT_G2_TABLE)
        g1_tbl = table_g1 or settings.LIBLOUIS_DEFAULT_G1_TABLE
        g2_tbl = table_g2 or settings.LIBLOUIS_DEFAULT_G2_TABLE
        if self._louis_module is not None:
            try:
                g1_ascii = self._louis_module.translateString([g1_tbl], text)
                g2_ascii = self._louis_module.translateString([g2_tbl], text)
                return BrailleTranslation(grade_1_ascii=g1_ascii, grade_2_ascii=g2_ascii, grade_1_unicode=self.ascii_to_unicode(g1_ascii), grade_2_unicode=self.ascii_to_unicode(g2_ascii), table_g1_used=g1_tbl, table_g2_used=g2_tbl)
            except Exception:
                pass
        try:
            g1_ascii = self._translate_g1_fallback(text)
            g2_ascii = self._translate_g2_fallback(text)
            return BrailleTranslation(grade_1_ascii=g1_ascii, grade_2_ascii=g2_ascii, grade_1_unicode=self.ascii_to_unicode(g1_ascii), grade_2_unicode=self.ascii_to_unicode(g2_ascii), table_g1_used=g1_tbl, table_g2_used=g2_tbl)
        except Exception as e:
            raise TranslationError(str(e)) from e


@dataclass
class TranslatedBlock:
    block: "DocumentBlock"
    translation: BrailleTranslation
    @property
    def grade_1_ascii(self) -> str: return self.translation.grade_1_ascii
    @property
    def grade_2_ascii(self) -> str: return self.translation.grade_2_ascii
    @property
    def grade_1_unicode(self) -> str: return self.translation.grade_1_unicode
    @property
    def grade_2_unicode(self) -> str: return self.translation.grade_2_unicode

def _translate_blocks(self: "BrailleTranslator", blocks: "Sequence[DocumentBlock]", grade: int = 1, table_g1: Optional[str] = None, table_g2: Optional[str] = None) -> List[TranslatedBlock]:
    results = []
    for block in blocks:
        try:
            braille = self.translate(text=block.clean_text(), table_g1=table_g1, table_g2=table_g2)
        except Exception:
            braille = BrailleTranslation(grade_1_ascii="", grade_2_ascii="", grade_1_unicode="", grade_2_unicode="", table_g1_used=table_g1 or settings.LIBLOUIS_DEFAULT_G1_TABLE, table_g2_used=table_g2 or settings.LIBLOUIS_DEFAULT_G2_TABLE)
        results.append(TranslatedBlock(block=block, translation=braille))
    return results

BrailleTranslator.translate_blocks = _translate_blocks
