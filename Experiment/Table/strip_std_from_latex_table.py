# strip_std_from_latex_table.py
# Usage:
#   python strip_std_from_latex_table.py table.tex table_mean_only.tex

import re
import sys
from pathlib import Path

# Match: $93.98{\scriptstyle \pm 0.39}$  (allow spaces, allow - sign)
STD_CELL_RE = re.compile(
    r"\$\s*([+-]?\d+(?:\.\d+)?)\s*\{\s*\\scriptstyle\s*\\pm\s*[+-]?\d+(?:\.\d+)?\s*\}\s*\$"
)

def strip_std(tex: str) -> str:
    # Replace std-format cells with just the mean number (no $...$)
    tex = STD_CELL_RE.sub(r"\1", tex)

    # (Optional) also handle cases like $0.00 \pm 0.00$ without \scriptstyle
    tex = re.sub(
        r"\$\s*([+-]?\d+(?:\.\d+)?)\s*\\pm\s*[+-]?\d+(?:\.\d+)?\s*\$",
        r"\1",
        tex
    )

    return tex

def main():
    if len(sys.argv) < 2:
        print("Usage: python strip_std_from_latex_table.py <in.tex> [out.tex]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path.with_suffix(".mean_only.tex")

    tex = in_path.read_text(encoding="utf-8")
    out = strip_std(tex)
    out_path.write_text(out, encoding="utf-8")

    print(f"Saved mean-only LaTeX to: {out_path}")

if __name__ == "__main__":
    main()