# -*- coding:utf-8 -*-

import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str, required=True)
	parser.add_argument('-o', '--out', type=str, default=None)

	args = parser.parse_args()

	with open(args.file, 'r') as f:
		data = f.read().split()

	# 同じ文字のみが連続するデータを省く
	for i, d in enumerate(data):
		if len(d) > 1 and len(set(d)) == 1:
			data[i] = d[0]

	data = sorted(list(set(data)))


	ret = ''
	for d in data:
		ret += d + '\n'

	outfile = args.file if args.out is None else args.out

	with open(outfile, 'w') as f:
		f.write(ret)

