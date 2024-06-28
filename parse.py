import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字') #也可以直接()，不用description
parser.add_argument('integers', type=str, help='传入的数字')

args = parser.parse_args() #解析方法

#获得传入的参数
print(args)