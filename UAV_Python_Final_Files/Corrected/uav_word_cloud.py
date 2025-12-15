from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Simplified extracted text from the LaTeX table
text = """
ISAC-UAV RIS CUs Altitude Speed Power Rician Noise QoS SNR DRL Learning Rate 
Actor Critic Replay Buffer Batch Size Gamma Soft Update Duration Time Slot 
Channel Model Path Loss Antennas Elements Environment Security Target 
Optimization Jamming UAV Position Fixed Episode
"""

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Simulation Parameters and System Model", fontsize=14)
plt.tight_layout()
plt.show()
