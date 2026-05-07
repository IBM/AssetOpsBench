import matplotlib.pyplot as plt
import numpy as np

dimensions = [
    "Task\ncompletion",
    "Data retrieval\naccuracy",
    "Generalized result\nverification",
    "Agent sequence\ncorrectness",
    "Clarity and\njustification",
    "Hallucination\nrate\n*lower rates desired",
]

thinking_off = [62, 100, 65, 88, 61, 12]
thinking_on  = [78, 100, 75, 88, 92,  5]

x = np.arange(len(dimensions))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars_off = ax.bar(x - bar_width / 2, thinking_off, bar_width,
                  label="Thinking off", color="#5A616C", edgecolor="white")
bars_on  = ax.bar(x + bar_width / 2, thinking_on,  bar_width,
                  label="Thinking on",  color="#BA5419FF", edgecolor="white")

ax.bar_label(bars_off, fmt="%d%%", padding=3, fontsize=9)
ax.bar_label(bars_on,  fmt="%d%%", padding=3, fontsize=9)

ax.set_xlabel("Evaluation Dimension", fontsize=11)
ax.set_ylabel("Accuracy Rate (%)", fontsize=11)
ax.set_title("LLM-as-Judge Accuracy per Dimension", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(dimensions, fontsize=9)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("llm_judge_accuracy.png", dpi=150)
plt.show()