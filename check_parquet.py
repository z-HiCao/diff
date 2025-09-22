import pyarrow.parquet as pq
import os
import argparse

def check_parquet_file(file_path):
    print(f"Checking file: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 读取 Parquet 文件的元数据
    try:
        parquet_file = pq.ParquetFile(file_path)
        
        # 获取并打印列名
        columns = parquet_file.schema.names
        print("Columns found in the Parquet file:")
        for column in columns:
            print(f"- {column}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description="Check columns in a Parquet file.")
    parser.add_argument("file_path", type=str, help="Path to the Parquet file.")
    args = parser.parse_args()
    
    check_parquet_file(args.file_path)