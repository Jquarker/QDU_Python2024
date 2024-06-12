import jieba
from wordcloud import WordCloud

with open('24_3/关于实施乡村振兴的意见.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用jieba进行中文分词
words = jieba.cut(text)
words = ' '.join(words)

wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                      width=800, height=400,
                      background_color='white').generate(words)

# 保存词云图
wordcloud.to_file('24_3/wordcloud.png')

# 显示词云图
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
