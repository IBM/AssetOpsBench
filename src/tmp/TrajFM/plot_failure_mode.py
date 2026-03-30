import pandas as pd
import plotly.express as px

# Data
data = {
    "cluster": [
        1, 6, 2, 4, 4, 2, 3, 1, 2, 5, 4, 1, 0, 3, 1, 5, 1, 2, 1, 6, 2, 4, 4, 2
    ],
    "failure_mode": [
        "Inadequate Error Handling",
        "Insufficient File Format Support",
        "Lack of Final Answer",
        "Lack of Adaptive Learning",
        "Lack of Adaptive Learning",
        "Lack of Final Answer",
        "Inconsistent Data Retrieval",
        "Inadequate Error Handling",
        "Lack of Final Answer",
        "Insufficient Feedback",
        "Lack of Adaptive Learning",
        "Inadequate Error Handling",
        "Inadequate Finalization",
        "Inconsistent Data Retrieval",
        "Inadequate Error Handling",
        "Insufficient Feedback",
        "Inadequate Error Handling",
        "Lack of Final Answer",
        "Inadequate Error Handling",
        "Insufficient File Format Support",
        "Lack of Final Answer",
        "Lack of Adaptive Learning",
        "Lack of Adaptive Learning",
        "Lack of Final Answer"
    ],
    "title": [
        "Inadequate Error Handling",
        "Insufficient File Format Support",
        "Lack of Final Answer",
        "Inadequate Contextual Understanding",
        "Lack of Adaptive Learning",
        "Lack of Final Answer",
        "Inconsistent Data Retrieval",
        "Lack of Error Handling",
        "Lack of Final Answer",
        "Insufficient Feedback",
        "Inadequate Sensor Relevancy Mapping",
        "Insufficient Failure Mode Documentation",
        "Inadequate Finalization",
        "Redundant Data Retrieval",
        "Insufficient Data Handling",
        "Lack of Feedback Mechanism",
        "Inability to Handle Missing Data",
        "Lack of Alternative Solutions",
        "Inadequate Error Handling",
        "Insufficient File Format Support",
        "Lack of Final Answer",
        "Inadequate Contextual Understanding",
        "Lack of Adaptive Learning",
        "Lack of Final Answer"
    ]
}

df = pd.DataFrame(data)

# Sunburst plot
fig = px.sunburst(
    df,
    path=['cluster', 'failure_mode', 'title'],  # hierarchy
    values=None,  # size determined automatically (count of rows)
    color='cluster',  # color by cluster
    color_continuous_scale='Viridis',
    title="Hierarchical Visualization of Failure Modes"
)

# Make it interactive and visually appealing
fig.update_traces(textinfo='label+percent entry', hoverinfo='label+value+percent parent')
fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))

# Save as HTML for interactivity or PNG for static image
fig.write_html("failure_modes_sunburst.html")  # interactive
fig.write_image("failure_modes_sunburst.png", scale=2)  # static high-res image
