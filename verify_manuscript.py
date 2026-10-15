import re
import sys
import os

def check_latex_file(tex_path, bib_path):
    print(f"Checking LaTeX file: {tex_path}")
    errors = []
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Environment balance
    begins = re.findall(r"\\begin\{([a-zA-Z0-9*]+)\}", content)
    ends = re.findall(r"\\end\{([a-zA-Z0-9*]+)\}", content)
    
    begin_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    end_counts = {}
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1
        
    mismatches = []
    for env, count in begin_counts.items():
        if end_counts.get(env, 0) != count:
            mismatches.append(f"Environment '{env}': {count} begins vs {end_counts.get(env, 0)} ends")
    for env, count in end_counts.items():
        if env not in begin_counts:
            mismatches.append(f"Environment '{env}': 0 begins vs {count} ends")

    if mismatches:
        print("  [ERROR] Environment mismatches found:")
        for m in mismatches:
            print(f"    - {m}")
        errors.append("Environment balance error")
    else:
        print(f"  [OK] All {len(begins)} LaTeX environments perfectly balanced.")

    # 2. Check citations
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib_content))

    citations = re.findall(r"\\cite\{([^}]+)\}", content)
    used_keys = set()
    for c in citations:
        for k in c.split(","):
            k = k.strip()
            if k:
                used_keys.add(k)

    missing_keys = used_keys - bib_keys
    if missing_keys:
        print(f"  [ERROR] Missing bibtex keys: {missing_keys}")
        errors.append("Missing bibtex keys")
    else:
        print(f"  [OK] All {len(used_keys)} cited keys exist in {bib_path}.")

    # 3. Check labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", content))
    missing_labels = refs - labels
    if missing_labels:
        print(f"  [ERROR] Missing labels for refs: {missing_labels}")
        errors.append("Missing labels for refs")
    else:
        print(f"  [OK] All {len(refs)} references have corresponding labels.")

    # 4. Check Title length in LaTeX
    title_match = re.search(r"\\title\{([^}]+)\}", content)
    if title_match:
        title = title_match.group(1).strip()
        title_words = title.split()
        print(f"  LaTeX Title ({len(title_words)} words): '{title}'")
        if 10 <= len(title_words) <= 15:
            print("  [OK] LaTeX Title length is within 10-15 words.")
        else:
            print(f"  [ERROR] LaTeX Title length is {len(title_words)} words (expected 10-15).")
            errors.append("Title word count error")

    # 5. Check Abstract in LaTeX
    abstract_match = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abs_words = abstract.split()
        paras = [p for p in abstract.split("\n\n") if p.strip()]
        print(f"  LaTeX Abstract ({len(abs_words)} words, {len(paras)} paragraph(s))")
        if 150 <= len(abs_words) <= 250:
            print("  [OK] LaTeX Abstract word count is within 150-250 words.")
        else:
            print(f"  [ERROR] LaTeX Abstract word count is {len(abs_words)} (expected 150-250).")
            errors.append("Abstract word count error")
        if len(paras) == 1:
            print("  [OK] LaTeX Abstract is exactly one unstructured paragraph.")
        else:
            print(f"  [ERROR] LaTeX Abstract has {len(paras)} paragraphs (expected exactly 1).")
            errors.append("Abstract paragraph count error")

    # 6. Check Conclusion in LaTeX
    conclusion_match = re.search(r"\\section\{Conclusion\}.*?\\label\{sec:conclusion\}(.+?)(?=\\section\*\{Declarations\}|\\section\*\{Acknowledgments\}|\\bibliographystyle|\Z)", content, re.DOTALL)
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip()
        clean_conclusion = re.sub(r"%.*", "", conclusion).strip()
        paras = [p for p in clean_conclusion.split("\n\n") if p.strip()]
        concl_words = clean_conclusion.split()
        print(f"  LaTeX Conclusion ({len(concl_words)} words, {len(paras)} paragraph(s))")
        if 150 <= len(concl_words) <= 200:
            print("  [OK] LaTeX Conclusion word count is within 150-200 words.")
        else:
            print(f"  [ERROR] LaTeX Conclusion word count is {len(concl_words)} (expected 150-200).")
            errors.append("Conclusion word count error")
        if len(paras) == 1:
            print("  [OK] LaTeX Conclusion is exactly one paragraph.")
        else:
            print(f"  [ERROR] LaTeX Conclusion has {len(paras)} paragraphs (expected exactly 1).")
            errors.append("Conclusion paragraph count error")

    # 7. Check Roadmap sentence in LaTeX
    expected_roadmap = "The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 describes the proposed methodology. Section 4 presents the experimental results. Section 5 discusses the findings and limitations, and Section 6 concludes."
    if expected_roadmap in content:
        print("  [OK] LaTeX Exact roadmap sentence found.")
    else:
        print("  [ERROR] LaTeX Exact roadmap sentence missing or modified.")
        errors.append("Roadmap sentence error")

    # 8. Check Paragraph 4 (Objective in one sentence) in LaTeX
    intro_match = re.search(r"\\section\{Introduction\}.*?\\label\{sec:introduction\}(.+?)\\begin\{itemize\}", content, re.DOTALL)
    if intro_match:
        intro_text = intro_match.group(1).strip()
        intro_paras = [p.strip() for p in intro_text.split("\n\n") if p.strip()]
        if len(intro_paras) >= 4:
            p4 = intro_paras[3]
            # split sentences by period followed by space
            p4_sentences = [s.strip() for s in re.split(r"\.\s+", p4) if s.strip()]
            print(f"  LaTeX Intro Paragraph 4 sentence count: {len(p4_sentences)}")
            if len(p4_sentences) == 1:
                print("  [OK] LaTeX Paragraph 4 is exactly one sentence.")
            else:
                print(f"  [WARNING] LaTeX Paragraph 4 has {len(p4_sentences)} sentences (expected exactly 1).")
                errors.append("Intro Paragraph 4 sentence count warning")

    # 9. Check Table scaling in LaTeX
    tables = re.findall(r"\\begin\{table[*]?\}(.+?)\\end\{table[*]?\}", content, re.DOTALL)
    print(f"  Total LaTeX tables found: {len(tables)}")
    for i, t in enumerate(tables, 1):
        has_resize = "\\resizebox" in t or "tabularx" in t or "p{" in t
        if has_resize:
            print(f"    [OK] Table {i} has width bounds (resizebox/tabularx/p{{...}}).")
        else:
            print(f"    [ERROR] Table {i} lacks width constraints!")
            errors.append(f"Table {i} lacks width constraints")

    # 10. Check Figure files existence
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)
    for fig in figs:
        fig_path = os.path.join("plots", os.path.basename(fig))
        if os.path.exists(fig_path):
            print(f"    [OK] LaTeX figure image exists: {fig_path}")
        else:
            print(f"    [ERROR] LaTeX figure image missing: {fig_path}")
            errors.append(f"Missing figure image {fig_path}")

    return len(errors) == 0

def check_markdown_file(md_path):
    print(f"\nChecking Markdown file: {md_path}")
    errors = []
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check Title length
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        title_words = title.split()
        print(f"  Markdown Title ({len(title_words)} words): '{title}'")
        if 10 <= len(title_words) <= 15:
            print("  [OK] Markdown Title length is within 10-15 words.")
        else:
            print(f"  [ERROR] Markdown Title length is {len(title_words)} words (expected 10-15).")
            errors.append("Title length error")

    # Check Abstract
    abstract_match = re.search(r"## Abstract\s+(.+?)(?=\n\n\*\*Keywords\*\*|\n\n\*\*Index Terms\*\*|\n\n---)", content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        paras = [p for p in abstract.split("\n\n") if p.strip()]
        abs_words = abstract.split()
        print(f"  Markdown Abstract ({len(abs_words)} words, {len(paras)} paragraph(s))")
        if 150 <= len(abs_words) <= 250:
            print("  [OK] Markdown Abstract word count is within 150-250 words.")
        else:
            print(f"  [ERROR] Markdown Abstract word count is {len(abs_words)} (expected 150-250).")
            errors.append("Abstract word count error")
        if len(paras) == 1:
            print("  [OK] Markdown Abstract is exactly one unstructured paragraph.")
        else:
            print(f"  [ERROR] Markdown Abstract has {len(paras)} paragraphs (expected exactly 1).")
            errors.append("Abstract paragraph count error")

    # Check Conclusion
    conclusion_match = re.search(r"## (?:Section 6\. )?Conclusion\s+(.+?)(?=\n\n## Declarations|\n\n## Acknowledgments|\n\n---|\Z)", content, re.DOTALL)
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip()
        paras = [p for p in conclusion.split("\n\n") if p.strip()]
        concl_words = conclusion.split()
        print(f"  Markdown Conclusion ({len(concl_words)} words, {len(paras)} paragraph(s))")
        if 150 <= len(concl_words) <= 200:
            print("  [OK] Markdown Conclusion word count is within 150-200 words.")
        else:
            print(f"  [ERROR] Markdown Conclusion word count is {len(concl_words)} (expected 150-200).")
            errors.append("Conclusion word count error")
        if len(paras) == 1:
            print("  [OK] Markdown Conclusion is exactly one paragraph.")
        else:
            print(f"  [ERROR] Markdown Conclusion has {len(paras)} paragraphs (expected exactly 1).")
            errors.append("Conclusion paragraph count error")

    # Check Roadmap sentence
    expected_roadmap = "The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 describes the proposed methodology. Section 4 presents the experimental results. Section 5 discusses the findings and limitations, and Section 6 concludes."
    if expected_roadmap in content:
        print("  [OK] Markdown Exact roadmap sentence found.")
    else:
        print("  [ERROR] Markdown Exact roadmap sentence missing or modified.")
        errors.append("Roadmap sentence error")

    # Check Paragraph 4 of Introduction
    intro_match = re.search(r"## Section 1\. Introduction\s+(.+?)(?=\n- \*\*|\n\nThe principal contributions)", content, re.DOTALL)
    if intro_match:
        intro_text = intro_match.group(1).strip()
        intro_paras = [p.strip() for p in intro_text.split("\n\n") if p.strip()]
        if len(intro_paras) >= 4:
            p4 = intro_paras[3]
            p4_sentences = [s.strip() for s in re.split(r"\.\s+", p4) if s.strip()]
            print(f"  Markdown Intro Paragraph 4 sentence count: {len(p4_sentences)}")
            if len(p4_sentences) == 1:
                print("  [OK] Markdown Paragraph 4 is exactly one sentence.")
            else:
                print(f"  [WARNING] Markdown Paragraph 4 has {len(p4_sentences)} sentences (expected exactly 1).")
                errors.append("Intro Paragraph 4 sentence count warning")

    # Check Algorithm 1 presence
    if "Algorithm 1" in content:
        print("  [OK] Markdown Algorithm 1 present.")
    else:
        print("  [ERROR] Markdown Algorithm 1 missing!")
        errors.append("Algorithm 1 missing in markdown")

    # Check required headings
    required_headings = [
        "## Section 1. Introduction",
        "## Section 2. Literature Review",
        "### 2.1 Deep Sequential Models for Vehicle Trajectory Forecasting",
        "### 2.2 Federated Learning in Cooperative Intelligent Transportation",
        "### 2.3 Non-IID Heterogeneity and Personalized Federated Learning",
        "### 2.4 Research Gap",
        "## Section 3. Methodology",
        "### 3.1 Problem formulation",
        "### 3.2 Data / materials",
        "### 3.3 Proposed approach",
        "### 3.4 Experimental setup",
        "## Section 4. Results",
        "### 4.1 Trajectory Prediction Benchmark",
        "### 4.2 Controlled Heterogeneity Resilience",
        "### 4.3 Privacy Attack Auditing & Spatial Fairness",
        "## Section 5. Discussion",
        "## Section 6. Conclusion",
        "## Declarations"
    ]
    missing_headings = [h for h in required_headings if h.lower() not in content.lower()]
    if missing_headings:
        print(f"  [ERROR] Missing required headings: {missing_headings}")
        errors.append("Missing required headings")
    else:
        print(f"  [OK] All {len(required_headings)} required headings present.")

    # Check Table 1 headers
    if "| Study | Approach | Dataset / Setting | Limitation |" in content:
        print("  [OK] Markdown Table 1 headers match specification exactly.")
    else:
        print("  [WARNING] Markdown Table 1 headers mismatch.")
        errors.append("Table 1 headers mismatch")

    # Check Figure images
    fig_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
    for fig in fig_links:
        if os.path.exists(fig):
            print(f"    [OK] Markdown figure image exists: {fig}")
        else:
            print(f"    [ERROR] Markdown figure image missing: {fig}")
            errors.append(f"Missing figure image {fig}")

    return len(errors) == 0

if __name__ == "__main__":
    tex_ok = check_latex_file("journal_paper.tex", "references.bib")
    md_ok = check_markdown_file("JOURNAL_PAPER_MANUSCRIPT.md")
    if tex_ok and md_ok:
        print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")
        sys.exit(0)
    else:
        print("\n>>> VERIFICATION FAILED! <<<")
        sys.exit(1)
