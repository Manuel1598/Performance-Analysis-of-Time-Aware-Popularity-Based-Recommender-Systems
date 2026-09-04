"""Generate thesis tables from audited final results, never from tuning scores."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recbole_results/final_analysis"
DEST = ROOT / "overleaf-thesis-project/tables/final"
NAMES = {"adressa_recbole_sample": "Adressa", "globo_recbole_sample": "Globo", "yoochoose_recbole_sample": "Yoochoose", "amazon_recbole": "Amazon", "movielens_recbole": "MovieLens"}
ORDER = ["MostPop", "RecentPop", "DecayPop", "BPR", "VS-KNN", "VSTAN", "GRU4Rec"]


def table(filename, caption, label, headers, spec, rows, note=""):
    lines = [r"\begin{table}[htbp]", r"\centering\small", "\\caption{"+caption+"}", "\\label{"+label+"}", r"\begin{tabular}{@{}"+spec+r"@{}}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize "+note+r"\end{minipage}"]
    lines += [r"\end{table}"]
    (DEST / filename).write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE / "final_results.csv")
    efficiency = frame[["dataset", "model", "execution_host", "mrr@10", "runtime_seconds"]].copy()
    efficiency["mrr_per_minute"] = efficiency["mrr@10"] * 60 / efficiency.runtime_seconds
    efficiency["pc2_nondominated"] = pd.NA
    pareto_rows = []
    for dataset, group in efficiency[efficiency.execution_host == "pc2"].groupby("dataset"):
        winners = []
        for index, r in group.iterrows():
            dominates = (group["mrr@10"] >= r["mrr@10"]) & (group.runtime_seconds <= r.runtime_seconds) & ((group["mrr@10"] > r["mrr@10"]) | (group.runtime_seconds < r.runtime_seconds))
            efficiency.loc[index, "pc2_nondominated"] = not dominates.any()
            if not dominates.any():
                winners.append(r.model)
        pareto_rows.append(NAMES[dataset]+" & "+", ".join(m for m in ORDER if m in winners)+r" \\")
    efficiency.to_csv(SOURCE / "runtime_efficiency.csv", index=False)
    table("runtime_pareto.tex", "Non-dominated PC2 models for MRR@10 and recorded total runtime.", "tab:final-pareto", ["Dataset", "Non-dominated models"], "lp{0.65\\textwidth}", pareto_rows, "A model is excluded if another PC2 model on the same dataset has at least as high MRR@10 and at most as much total runtime, with one strict improvement. VSTAN is not assessed across hosts. This is a descriptive frontier, not a repeated timing benchmark.")
    for scenario in ("topn", "session"):
        subset = frame[frame.scenario == scenario]
        quality, exposure, runtime = [], [], []
        for dataset, group in subset.groupby("dataset", sort=True):
            group = group.set_index("model").reindex([m for m in ORDER if m in set(group.model)]).reset_index()
            for rows, n in ((quality,7),(exposure,5),(runtime,5)):
                rows += [r"\multicolumn{"+str(n)+r"}{l}{\textit{"+NAMES[dataset]+r"}} \\"]
            best = group["mrr@10"].max()
            for _, r in group.iterrows():
                vals = [f"{r[m]:.4f}" for m in ("hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10")]
                if r["mrr@10"] == best:
                    vals[-1] = r"\textbf{"+vals[-1]+"}"
                quality.append(" & ".join([r.model]+vals)+r" \\")
                exposure.append(" & ".join([r.model, f"{100*r['coverage@10']:.3f}", f"{int(r['unique_recommended_items@10']):,}", f"{r['avg_recommendation_popularity@10']:,.1f}", f"{r['recommendation_frequency_gini@10']:.4f}"])+r" \\")
                runtime.append(" & ".join([r.model, "PC1" if r.model == "VSTAN" else "PC2", f"{r.runtime_seconds:.2f}", f"{r.evaluation_runtime_seconds:.2f}", f"{r.extra_metrics_runtime_seconds:.2f}"])+r" \\")
            quality.append(r"\addlinespace")
            exposure.append(r"\addlinespace")
            runtime.append(r"\addlinespace")
        name = "Top-N" if scenario == "topn" else "Session"
        table(f"{scenario}_quality.tex", f"{name} ranking quality on the held-out test partition (seed 42).", f"tab:final-{scenario}-quality", ["Model","Hit@5","Hit@10","NDCG@5","NDCG@10","MRR@5","MRR@10"], "lrrrrrr", quality, "Bold marks the highest MRR@10 within each dataset. All entries are decimal values, not percentages.")
        table(f"{scenario}_exposure.tex", f"{name} catalogue exposure at cutoff ten.", f"tab:final-{scenario}-exposure", ["Model", "Coverage (\\%)", "Items", "Avg. count", "Gini"], "lrrrr", exposure, "Items is the number of distinct recommended items. Avg. count is their mean training-target frequency, weighted by recommendation occurrences. Gini includes catalogue items never recommended.")
        table(f"{scenario}_runtime.tex", f"{name} recorded final-run times in seconds.", f"tab:final-{scenario}-runtime", ["Model", "Host", "Total", "Ranking", "Extra metrics"], "llrrr", runtime, "Total also includes data preparation, model construction, fitting, validation and checkpoint handling. Ranking and extra metrics are separate test-prediction passes. PC1 VSTAN times must not be ranked against PC2 times as a controlled hardware comparison.")
    chart = [r"\begin{figure}[htbp]", r"\centering", r"\begin{tikzpicture}", r"\begin{axis}[thesisbar,width=0.96\textwidth,height=6.8cm,bar width=6pt,ymin=0,ymax=0.34,ylabel={MRR@10},symbolic x coords={Adressa,Globo,Yoochoose},xtick=data,legend columns=3,legend style={at={(0.5,-0.18)},anchor=north,font=\footnotesize},enlarge x limits=0.25]"]
    colors = ["gray!35", "gray!65", "black!80", "blue!65", "cyan!65", "orange!85"]
    models = [m for m in ORDER if m != "BPR"]
    for model, color in zip(models, colors):
        data = frame[(frame.scenario == "session") & (frame.model == model)]
        coords = " ".join(f"({NAMES[r.dataset]},{r['mrr@10']:.4f})" for _, r in data.iterrows())
        chart.append(r"\addplot[fill="+color+r",draw=black!50] coordinates {"+coords+"};")
    chart += [r"\legend{"+",".join(models)+"}", r"\end{axis}", r"\end{tikzpicture}", r"\caption{Final session MRR@10 by dataset. Model rankings are compared within each dataset; there is no single winner across the three samples.}", r"\label{fig:final-session-mrr}", r"\end{figure}"]
    (DEST / "session_mrr_figure.tex").write_text("\n".join(chart)+"\n", encoding="utf-8")
    prefix_dir = SOURCE / "prefix_groups"
    expected = frame[frame.scenario == "session"]
    paths = [prefix_dir / f"{r.dataset}__{r.model}_groups.csv" for _,r in expected.iterrows()]
    if not all(p.exists() for p in paths):
        print("Main tables written; prefix tables await all 18 verified replays")
        return
    groups = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Same prefix/target ordering must be used by all models on a dataset.
    for dataset, model_rows in expected.groupby("dataset"):
        audits = [json.loads((prefix_dir / f"{dataset}__{r.model}.json").read_text()) for _,r in model_rows.iterrows()]
        if any(a["status"] != "success" for a in audits) or len({a["test_queries_sha256"] for a in audits}) != 1:
            raise ValueError(f"Test query fingerprint mismatch: {dataset}")
        sub = groups[groups.dataset == dataset]
        if sub.groupby("prefix_group").query_count.nunique().max() != 1:
            raise ValueError("Different prefix groups across models")
        rows = []
        for prefix in ("1", "2", "3", "4+"):
            g = sub[sub.prefix_group.astype(str) == prefix]
            count = int(g.iloc[0].query_count)
            if not count:
                continue
            rows += [r"\multicolumn{4}{l}{\textit{"+prefix+f" available clicks; {count:,} test queries"+r"}} \\"]
            for model in ORDER:
                if model not in set(g.model):
                    continue
                r = g[g.model == model].iloc[0]
                vals = [f"{r[m]:.4f}" if count else "--" for m in ("hit@10", "mrr@10", "ndcg@10")]
                rows.append(" & ".join([model]+vals)+r" \\")
            rows.append(r"\addlinespace")
        table(f"prefix_{NAMES[dataset].lower()}.tex", f"{NAMES[dataset]} test quality by the number of available input clicks.", f"tab:prefix-{NAMES[dataset].lower()}", ["Model", "Hit@10", "MRR@10", "NDCG@10"], "lrrr", rows)
    counts = groups[groups.model == "MostPop"].copy()
    rows = [" & ".join([NAMES[r.dataset], str(r.prefix_group), f"{int(r.query_count):,}", f"{100*r.query_share:.2f}"])+r" \\" for _,r in counts.iterrows()]
    table("prefix_counts.tex", "Composition of the session test partitions by available prefix length.", "tab:prefix-counts", ["Dataset", "Input clicks", "Test queries", "Share (\\%)"], "llrr", rows)
    groups.to_csv(SOURCE / "prefix_group_results.csv", index=False)
    print("All main and prefix tables written")


if __name__ == "__main__":
    main()
