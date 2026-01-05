import os
import re
from typing import List, Set
import unicodedata
from pyvi import ViTokenizer


URL_RE     = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
USER_RE    = re.compile(r'@[A-Za-z0-9_]+')
EMAIL_RE   = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
PHONE_RE   = re.compile(r'\b(?:\+?84|0)(?:[1-9]\d){8,10}\b')
MONEY_RE   = re.compile(r'(\d+[.,]?\d*)\s?(k|nghìn|ngan|ngàn|tr|triệu|m|đ|vnd)\b', re.IGNORECASE)
PERCENT_RE = re.compile(r'\b\d+[.,]?\d*\s?%\b')
NUMBER_RE  = re.compile(r'\b\d+[.,]?\d*\b')

ZW_NBSP_RE = re.compile(r'[\u200b\xa0]')
MULTI_SPACE= re.compile(r'\s+')
REPEAT_CHAR= re.compile(r'(.)\1{2,}', re.DOTALL)

# ====== Emoji/Emoticon Patterns ======
EMOJI_ALL_RE = re.compile(
    "["
    + "\U0001F600-\U0001F64F"  # Emoticons
    + "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    + "\U0001F680-\U0001F6FF"  # Transport & Map
    + "\U0001F1E0-\U0001F1FF"  # Flags
    + "\U00002702-\U000027B0"  # Dingbats
    + "\U000024C2-\U0001F251"  # Enclosed
    + "\U0001F926-\U0001F937"
    + "\U00010000-\U0010FFFF"  # Plane 1+
    + "\u200d" + "\u2640-\u2642" + "\u2600-\u2B55" + "\u23cf" + "\u23e9" + "\u231a" + "\u3030" + "\ufe0f"
    + "]",
    re.UNICODE
)

VS16_RE      = re.compile("\ufe0f")
SKIN_TONE_RE = re.compile("[" + "\U0001F3FB-\U0001F3FF" + "]")
ZWJ_RE       = re.compile("\u200d")

LAUGH_RE = re.compile(r"(=+\)+|\)+\)+|ha(?:ha)+|hi(?:hi)+|kkk+|hu+hu+|h(?:ì|í)+h(?:ì|í)+)", re.IGNORECASE)
EMO_POS_RE = re.compile(
    r"("
    r"[" r"\U0001F600-\U0001F606" r"\U0001F60A" r"\U0001F60D" r"\U0001F642" r"\U0001F929" r"\U0001F970" r"\U0001F618" r"\u263A" r"]"
    r"|(?:\u2764\ufe0f|\u2764)"
    r"|\U0001F44D|\U0001F44C"
    r"|:D|:d|:\)+|\^\^|=+\)+|:v"
    r")",
    re.UNICODE
)
EMO_NEG_RE = re.compile(
    r"("
    r"[" r"\U0001F61E" r"\U0001F61F" r"\U0001F620" r"\U0001F621" r"\U0001F622" r"\U0001F62D" r"\U0001F494" r"\U0001F641" r"\u2639" r"]"
    r"|\U0001F44E|:\(|=\(+|T_T|>\\.<"
    r")",
    re.UNICODE
)
EMO_NEU_RE = re.compile(r"[" r"\U0001F610" r"\U0001F611" r"\U0001F636" r"\U0001F914" r"\U0001F644" r"]", re.UNICODE)

PUNCT_KEEP = "!?…"
PUNCT_RE   = re.compile(rf"[^\w\s{PUNCT_KEEP}]")
EXCLAM_RE  = re.compile(r"!{2,}")
QUEST_RE   = re.compile(r"\?{2,}")

NEGS = {"không", "chẳng", "chả", "ko", "kh"}

# ====== Teencode ======
def load_teencode_regex_list(path="teencode.txt"):
    pairs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f.read().splitlines():
                if not ln.strip(): continue
                parts = ln.split("\t")
                if len(parts) >= 2:
                    src, dst = parts[0].strip(), parts[1].strip()
                    if src:
                        pairs.append((re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst))
    else:
        fallback = {
            "ko":"không","k":"không","kh":"không","hok":"không","hong":"không","hk":"không",
            "dc":"được","đc":"được","ms":"mới","ng":"người","trc":"trước","mik":"mình","vs":"với",
            "thik":"thích","bt":"bình_thường","ntn":"như_thế_nào","sp":"sản_phẩm","ok":"okay","oke":"okay",
            "vl":"rất","vkl":"rất"
        }
        for src, dst in fallback.items():
            pairs.append((re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst))
    return pairs

TEENCODE_REGEX = load_teencode_regex_list("teencode.txt")

# ====== Stopwords ======
def load_stopwords(path: str = "vietnamese_stopwords.txt") -> Set[str]:
    """Load stopwords from file, fallback to minimal set"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    else:
        return {
            'thì', 'là', 'mà', 'có', 'với', 'được', 'này', 'kia', 'ấy', 'vậy',
            'về', 'nên', 'sẽ', 'đã', 'cũng', 'vẫn', 'vào', 'ra', 'lên', 'xuống',
            'nào', 'ai', 'gì', 'nơi', 'hãy', 'đừng', 'chớ', 'rất', 'quá', 'lắm'
        }

STOPWORDS = load_stopwords()

# ====== Helpers ======
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))

def _squash_repeats(s: str) -> str:
    return REPEAT_CHAR.sub(r"\1\1", s)

def _replace_special_tokens(s: str) -> str:
    s = URL_RE.sub(' __url__ ', s)
    s = EMAIL_RE.sub(' __email__ ', s)
    s = PHONE_RE.sub(' __phone__ ', s)
    s = PERCENT_RE.sub(' __percent__ ', s)
    s = MONEY_RE.sub(' __money__ ', s)
    s = NUMBER_RE.sub(' __num__ ', s)
    s = USER_RE.sub(' __user__ ', s)
    return s

def _apply_teencode(s: str) -> str:
    for rgx, rep in TEENCODE_REGEX:
        s = rgx.sub(rep, s)
    return s

def _tag_emojis(s: str) -> str:
    t = normalize_emoji_variants(s)
    t = LAUGH_RE.sub(" __laugh__ ", t)
    t = EMO_POS_RE.sub(" __emo_pos__ ", t)
    t = EMO_NEG_RE.sub(" __emo_neg__ ", t)
    t = EMO_NEU_RE.sub(" __emo_neu__ ", t)
    return t

def normalize_emoji_variants(s: str) -> str:
    s = VS16_RE.sub("", s)
    s = SKIN_TONE_RE.sub("", s)
    s = ZWJ_RE.sub("", s)
    return s

def join_negation(text: str, window: int = 1) -> str:
    toks, out, i = text.split(), [], 0
    bad = {"__emo_pos__","__emo_neg__","__url__","__email__","__percent__","__money__","__num__","__user__"}
    while i < len(toks):
        w = toks[i]
        if w in NEGS and i + 1 < len(toks):
            nxt = toks[i+1]
            if nxt not in bad and not any(ch in nxt for ch in "!?."):
                out.append(w + "_" + nxt)
                i += 2
                continue
        out.append(w); i += 1
    return " ".join(out)

def _remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS or t.startswith('__')]

# ====== Main Normalization ======
def normalize_text(
    s: str,
    keep_all_emojis: bool = False,
    use_vitok: bool = True,
    remove_stopwords_flag: bool = False,
    join_negation_flag: bool = True
) -> str:
    # 1) Basic normalization
    s = _nfc(s)
    s = ZW_NBSP_RE.sub(' ', s)
    s = _replace_special_tokens(s)
    s = _squash_repeats(s)

    # 2) Teencode & Emoji tagging
    s = _apply_teencode(s)
    s = _tag_emojis(s)

    # 3) Lowercase + tokenize
    s = s.lower().strip()
    if use_vitok:
        s = ViTokenizer.tokenize(s)  # Sử dụng pyvi để tokenize

    # 4) Punctuation handling
    s = EXCLAM_RE.sub(" __exclam__ ", s)
    s = QUEST_RE.sub(" __quest__ ", s)
    s = s.replace("...", " __ellips__ ")
    s = PUNCT_RE.sub(" ", s)
    s = MULTI_SPACE.sub(' ', s).strip()

    # 5) Negation joining
    if join_negation_flag:
        s = join_negation(s, window=1)

    # 6) Stopword removal (optional)
    if remove_stopwords_flag:
        tokens = s.split()
        tokens = _remove_stopwords(tokens)
        s = " ".join(tokens)

    # 7) Remove remaining emojis (optional)
    if not keep_all_emojis:
        s = EMOJI_ALL_RE.sub(" ", s)
        s = MULTI_SPACE.sub(' ', s).strip()

    return s

