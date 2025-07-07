import os

def merge_and_deduplicate_txt_files(input_folder, output_file):
    """
    将指定文件夹下所有 .txt 文件的内容整合到一个文件中，并对每行进行去重。

    Args:
        input_folder (str): 包含多个 .txt 文件的文件夹路径。
        output_file (str): 整合并去重后内容的输出文件路径。
    """
    # 使用集合（set）来存储所有不重复的行
    unique_lines = set()
    
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
        return

    print(f"开始处理文件夹：'{input_folder}'")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理 .txt 文件
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            print(f"  - 正在读取文件：'{filepath}'")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped_line = line.strip() # 移除行首尾的空白字符，包括换行符
                        if stripped_line: # 确保非空行才添加
                            unique_lines.add(stripped_line)
            except Exception as e:
                print(f"    警告：读取文件 '{filepath}' 时发生错误：{e}")
                continue

    # 将去重后的内容写入输出文件
    print(f"\n开始写入去重后的内容到文件：'{output_file}'")
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 排序（可选）：如果希望输出文件中的行有序，可以取消下面这行的注释
            # sorted_lines = sorted(list(unique_lines))
            # for line in sorted_lines:
            #     f_out.write(line + '\n')
            
            # 不排序：直接写入，顺序不确定
            for line in unique_lines:
                f_out.write(line + '\n')
        print(f"成功将内容整合并去重到 '{output_file}'。共写入 {len(unique_lines)} 条不重复的行。")
    except Exception as e:
        print(f"错误：写入文件 '{output_file}' 时发生错误：{e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义输入文件夹和输出文件路径
    # 假设你的所有txt文件都在一个名为 'input_txt_files' 的文件夹里
    # 最终合并去重后的内容将保存到 'merged_deduplicated.txt'
    
    # 获取当前脚本的目录
    
    # 定义示例输入文件夹和输出文件路径
    input_folder_path = "data"
    output_file_path = os.path.join("./", "merged_deduplicated.txt")

    # 调用函数执行整合和去重
    merge_and_deduplicate_txt_files(input_folder_path, output_file_path)
