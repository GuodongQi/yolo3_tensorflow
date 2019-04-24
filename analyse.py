# coding=utf-8
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import os

video_path = "F:\\wzry\\三杀"
heros = os.listdir(video_path)

# heros_count = []
# for hero in heros:
#     for root, dirts, files in os.walk(os.path.join(video_path, hero)):
#         for j in range(len(files)):
#             heros_count.append(hero)
# heros_count = " ".join(heros_count)
# print(heros_count)
# font = r'C:\Windows\Fonts\simfang.ttf'
# wordcloud = WordCloud(
#     font_path=font,  # 如果是中文必须要添加这个，否则会显示成框框
#     background_color='white',
#     max_words=2000,
#     collocations=False
#     # stopwords={'to', 'of'}  # set or space-separated string
# ).generate_from_text(heros_count)


heros_count = {}
for hero in heros:
    for root, dirts, files in os.walk(os.path.join(video_path, hero)):
        heros_count[hero] = len(files)
print(heros_count)

font = r'C:\Windows\Fonts\simfang.ttf'
wordcloud = WordCloud(
    font_path=font,  # 如果是中文必须要添加这个，否则会显示成框框
    background_color='white',
    height=800,
    width=1000,
    max_words=1000,
    collocations=False,
    scale=0.5
    # stopwords={'to', 'of'}  # set or space-separated string
).generate_from_frequencies(heros_count)

plt.imshow(wordcloud)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴
# plt.show()  # 显示图片
plt.savefig("fra_wine.png", format="png")