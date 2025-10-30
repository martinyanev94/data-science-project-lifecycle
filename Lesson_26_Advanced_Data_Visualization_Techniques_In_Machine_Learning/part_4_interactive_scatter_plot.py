import plotly.express as px

fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Target', title='Interactive Scatter Plot of Features')
fig.show()
