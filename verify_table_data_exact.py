import sys
import docx
import re

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

with open('JOURNAL_PAPER_MANUSCRIPT.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

# Extract all 6 markdown tables
md_tables_raw = re.findall(r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)', md_text)
print(f"Extracted {len(md_tables_raw)} tables from markdown.")

md_tables = []
for raw in md_tables_raw:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    hdr = [c.strip() for c in lines[0].split('|')[1:-1]]
    data = []
    for l in lines[2:]: # skip separator
        row = [c.strip() for c in l.split('|')[1:-1]]
        data.append(row)
    md_tables.append((hdr, data))

# Extract data tables from docx (tables with > 2 columns)
doc = docx.Document('journal_paper.docx')
docx_tables = []
for t in doc.tables:
    if len(t.columns) > 2:
        hdr = [c.text.strip() for c in t.rows[0].cells]
        data = []
        for r in t.rows[1:]:
            row = [c.text.strip() for c in r.cells]
            data.append(row)
        docx_tables.append((hdr, data))

print(f"Extracted {len(docx_tables)} data tables from docx.")

def normalize_syms(v):
    v = v.replace('$^{\\dagger}$', '†').replace('\\dagger', '†')
    v = v.replace('$^{\\ddagger}$', '‡').replace('\\ddagger', '‡')
    v = v.replace('\\overline{\\mathrm{JSD}}', 'JSD̄').replace('\\overline{JSD}', 'JSD̄')
    v = v.replace('\\mathrm{JSD}', 'JSD')
    v = v.replace('\\pm', '±')
    v = v.replace('\\le', '≤')
    v = v.replace('\\ge', '≥')
    v = v.replace('\\times', '×')
    v = v.replace('\\sigma', 'σ')
    v = v.replace('\\epsilon', 'ε')
    v = v.replace('\\delta', 'δ')
    v = v.replace('\\mu', 'μ')
    v = v.replace('\\lambda', 'λ')
    v = v.replace('\\eta', 'η')
    v = v.replace('\\infty', '∞')
    v = v.replace('$-$', '−')
    v = v.replace('−', '-').replace('–', '-').replace('—', '-').replace('--', '-')
    v = re.sub(r'[*_$\\\\`]', '', v)
    v = v.replace('{', '').replace('}', '')
    v = ' '.join(v.split())
    return v

# Compare table by table
for t_idx in range(min(len(md_tables), len(docx_tables))):
    md_hdr, md_data = md_tables[t_idx]
    dx_hdr, dx_data = docx_tables[t_idx]
    
    print(f"\n--- Checking Table {t_idx+1} ---")
    print(f"  Markdown shape: {len(md_data)} rows x {len(md_hdr)} cols")
    print(f"  Docx shape:     {len(dx_data)} rows x {len(dx_hdr)} cols")
    
    # Check headers
    hdr_mismatches = []
    for c_idx, (mh, dh) in enumerate(zip(md_hdr, dx_hdr)):
        if normalize_syms(mh).lower() != normalize_syms(dh).lower():
            hdr_mismatches.append((c_idx, mh, dh))
    if hdr_mismatches:
        print(f"  Header mismatches: {hdr_mismatches}")
    else:
        print("  Headers match 100%!")
        
    # Check rows
    row_mismatches = []
    for r_idx in range(min(len(md_data), len(dx_data))):
        mr = md_data[r_idx]
        dr = dx_data[r_idx]
        for c_idx in range(min(len(mr), len(dr))):
            mv = normalize_syms(mr[c_idx])
            dv = normalize_syms(dr[c_idx])
            if mv != dv:
                row_mismatches.append((r_idx, c_idx, mr[c_idx], dr[c_idx]))
                
    if row_mismatches:
        print(f"  Cell mismatches ({len(row_mismatches)}):")
        for rm in row_mismatches[:10]:
            print(f"    Row {rm[0]}, Col {rm[1]}: MD='{rm[2]}' vs DOCX='{rm[3]}'")
    else:
        print("  ALL data cells match 100% exactly with markdown!")
