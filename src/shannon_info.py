# shannon_info.py
#
# This program is an implementation of the formula given in 
# Russell (2003) p. 659, which is from Shannon and Weaver (1949),
# to determine the amount of information needed given counts
# of the possible classifications.
#
# corey.abshire@gmail.com
# 2012-09-03

import sys
import math

v = [float(a) for a in sys.argv[1:]]
s = sum(v)
print 'sum:', s
i = 0.0
for n in v:
    i -= (n / s) * math.log(n / s, 2)
    print 'i:', n, i
print 'final i:', i
