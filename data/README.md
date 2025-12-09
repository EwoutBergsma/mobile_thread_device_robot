`uart_devttyacm0_5777002735_20251203_171946.log`
`uart_devttyacm1_578e013641_20251203_171946.log`
First seemingly succesful data captured from two ESP32 devices, running the same one with PPS (1 minute, 0 backoff) and the other without. Improvements to be made:
- There is no pinging
- There is no current measurement

__________

`uart_devttyacm1_5777002735_20251204_175233.log`
`uart_devttyacm0_578e013641_20251204_175233.log`
Now with pinging every second

__________

`uart_devttyacm0_5777002735_20251208_190228.log`
`uart_devttyacm1_578e013641_20251208_190228_PPS.log`
Tried doing all routers and end devices to -13 dbm
Mistake found, end devices did not understand the txpower command (see beginning of logs)
__________

`uart_devttyacm0_5777002735_20251209_111508.log`
`uart_devttyacm0_5777002735_20251209_111819.log`
`uart_devttyacm1_578e013641_20251209_111508_PPS.log`
`uart_devttyacm1_578e013641_20251209_111832_PPS.log`

These appear to be good results with every device properly set to -13 dbm txpower. Weirdly this became two separate logs, not sure why. For this reason perhaps we should try to also periodially check what the current transmit power of the device is.
Also, this was a relatively short run.
__________
