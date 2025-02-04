This code combines an LSTM and a Convolutional Neural Network (CNN) to classify normalized 2D market data into three categories: 
longs, shorts, and no positions. The data is normalized using methods like Z-score normalization, Min-Max scaling, and others, 
but due to the high noise in market data, achieving reliable predictions remains 
challenging. Despite multiple attempts, the models struggled to surpass a ~45% success rate 
(where random guessing would achieve ~33%). However, other methods such as reinforcement learning and 
machine learning techniques like genetic algorithms have shown better results in similar tasks. Sensitive information, 
including proprietary datasets, has been removed from this version.