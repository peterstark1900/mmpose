import json

# 假设你的 JSON 文件名为 'Fish-Tracker-1001-Test-detection.json'
input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test-detection.json'
output_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test-detection-with-score.json'

# 读取 JSON 文件
with open(input_file, 'r') as f:
    data = json.load(f)

# 为每个 bbox 添加 score 属性
for item in data:
    item['score'] = 0.95  # 你可以根据需要设置 score 的值

# 将修改后的数据写回到 JSON 文件中
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"已为每个 bbox 添加 score 属性，并保存到 {output_file}")