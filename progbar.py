import SpinWaveToolkit as swt
from time import sleep


niter = 55
pb = swt.ProgressBar(niter)
for i in range(niter):
    sleep(0.1)
    pb.next()
