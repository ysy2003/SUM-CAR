#!/usr/bin/env python3
import json, argparse

EXAMPLES = [
  {
    'id': 'fin+code:operating_margin',
    'prompt': (
      "Given the earnings excerpt below, first extract last quarter's revenue and operating income, "
      "then write Python code to compute operating margin in percent.\n\n"
      "Context:\nRevenue: $12.5M; Operating Income: $1.2M.\n\nAnswer:"
    ),
    'gold_numbers': ['12.5', '1.2'],
    'tests': 'assert abs((1.2/12.5)*100 - 9.6) < 1e-6\nprint("passed")\n'
  },
  {
    'id': 'math+code:gsm2py',
    'prompt': (
      'Solve the problem and then output Python code that computes the same answer.\n\n'
      'Problem: If a car travels 60 miles in 1.5 hours, what is the average speed in mph?\n\nAnswer:'
    ),
    'gold_numbers': ['60', '1.5', '40'],
    'tests': 'def _f():\n    return 60/1.5\nassert abs(_f()-40.0) < 1e-6\nprint("passed")\n'
  }
]

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  with open(args.out, 'w', encoding='utf-8') as f:
    for row in EXAMPLES:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  print('composite set saved to', args.out)
