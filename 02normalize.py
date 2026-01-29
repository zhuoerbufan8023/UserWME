# -*- ecoding: utf-8 -*-
# @ModuleName: normize
# @Function: 
# @Author: Wang Zhuo
# @Time: 2024-06-11 16:39
import pandas as pd

# 读取Excel文件
df = pd.read_excel('./data/result_pivot.xlsx')

# 定义归一化函数
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

# 对指定列进行归一化处理
columns_to_normalize = ["satisfaction","functional","explanatory","aesthetic","reliability"]
for column in columns_to_normalize:
    df[f'{column}_normalized'] = (normalize_column(df[column])*10).round(2)

# 显示结果
print(df)

# 保存结果到新的Excel文件
df.to_excel('./data/data_normalized.xlsx', index=False)
