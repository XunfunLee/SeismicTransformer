import h5py
import os
from collections import Counter

def get_damage_state_counts(h5_file_path):
    with h5py.File(h5_file_path, 'r') as file:
        # 确保数据是一维数组
        damage_states = file['Blg_Damage_State'][:]
        if damage_states.ndim > 1:
            # 如果数据集是多维的，将其转换为一维
            damage_states = damage_states.flatten()
        # 计数损伤状态
        return Counter(damage_states)

def main(folder_path, output_file):
    # 用来统计各个损伤等级的计数器
    damage_state_counter = Counter()

    # 遍历文件夹下的所有h5文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            # 更新损伤等级计数器
            damage_state_counter.update(get_damage_state_counts(file_path))

    # 打开文件准备写入结果
    with open(output_file, 'w') as f:
        # 写入损伤等级及其数量到文件中
        for state, count in sorted(damage_state_counter.items()):
            f.write(f'Damage State {state}: {count}\n')
        print(f'Results have been saved to {output_file}')

    # 返回损伤等级计数器以供后续处理
    return damage_state_counter

if __name__ == '__main__':
    folder_path = 'D:\\SeismicTransformerData\\SeT-4.0'  # 替换为你的hdf5文件夹路径
    output_file = 'Num_samples_class.txt'  # 要写入的文件名
    damage_state_counts = main(folder_path, output_file)