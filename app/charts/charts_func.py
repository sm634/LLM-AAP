import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle


# Create donut chart function
def donut_chart(data, chart_title, legend_col, num_col='Count'):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(data[num_col], labels=None,
           autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))

    # Draw a white circle at the center to create a donut chart
    centre_circle = Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title(chart_title, fontdict={'family': 'serif', 'color': 'black', 'size': 20})
    # add legend
    plt.legend(data[legend_col], loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))
