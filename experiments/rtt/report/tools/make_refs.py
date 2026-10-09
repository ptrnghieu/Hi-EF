# Danh mục tài liệu tham khảo theo quy định trình bày KLTN của Trường ĐH Công nghệ (Phụ lục 04):
# xếp theo ngôn ngữ, trong mỗi ngôn ngữ theo ABC họ của tác giả đầu tiên; [STT] Tác giả, "Tên bài báo", Tên tạp chí/
# kỷ yếu (nghiêng), tập, số, năm, pp. trang.  Sách: Tác giả, Tên sách (nghiêng), Nhà xuất bản, năm.
# Usage: python make_refs.py references.bib cover/tailieuthamkhao.tex
import re, sys

src = open(sys.argv[1], encoding='utf-8').read()
entries = []
for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', src, re.S):
    typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
    f = {}
    for fm in re.finditer(r'(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', body):
        f[fm.group(1).lower()] = re.sub(r'\s+', ' ', fm.group(2)).strip()
    entries.append((typ, key, f))


def split_name(n):
    last, first = [x.strip() for x in n.split(',', 1)] if ',' in n else (n.split()[-1], ' '.join(n.split()[:-1]))
    return last, first


def initials(first):
    out = []
    for tok in first.replace('-', ' -').split():
        if tok.startswith('{\\'):                              # e.g. {\L}ukasz
            out.append(tok[:tok.index('}') + 1] + '.')
        elif tok.startswith('-'):
            out[-1] = out[-1] + '-' + tok[1] + '.'
        else:
            out.append(tok[0] + '.')
    return '~'.join(out)


def authors(s):
    names = [split_name(a) for a in s.split(' and ')]
    fmt = [f"{initials(fi)}~{la}" if fi else la for la, fi in names]
    return fmt[0] if len(fmt) == 1 else ', '.join(fmt[:-1]) + ' and ' + fmt[-1], names


def natlabel(names, year):
    la = [n[0] for n in names]
    short = la[0] + (' et~al.' if len(la) > 2 else (' and ' + la[1] if len(la) == 2 else ''))
    return f"{short}({year})"


def sortkey(e):
    la, fi = split_name(e[2]['author'].split(' and ')[0])
    return (re.sub(r'[{}\\"\']', '', la).lower(), e[2].get('year', ''))


lines = [r"\begin{thebibliography}{99}", r"\section*{Tiếng Anh}"]
for typ, key, f in sorted(entries, key=sortkey):
    au, names = authors(f['author'])
    year, title = f.get('year', ''), f['title']
    parts = [au]
    if typ == 'book':
        parts += [r"\emph{" + title + "}", f.get('publisher', ''), year]
    else:
        venue = f.get('journal') or f.get('booktitle', '')
        parts += ["``" + title + "''", r"\emph{" + venue + "}"]
        if 'volume' in f:
            parts.append('Vol.~' + f['volume'])
        if 'number' in f:
            parts.append('No.~' + f['number'])
        parts.append(year)
        if 'pages' in f:
            parts.append('pp.~' + f['pages'])
    lines.append(f"\\bibitem[{natlabel(names, year)}]{{{key}}} " + ', '.join(p for p in parts if p) + '.')
lines.append(r"\end{thebibliography}")
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(entries), 'references written')
