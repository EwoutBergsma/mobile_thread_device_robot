# 


```cmd

    ssh fieldlab@10.149.24.26


// to start the experiment:
// nohup python3 wander.py > wander.log 2>&1 &
// nohup python3 esp32_uart_logger.py > esp32_logger.log 2>&1 &

nohup python3 wander.py > wander.log 2>&1 & nohup python3 esp32_uart_logger.py > esp32_uart_logger.log 2>&1 &

// to stop the node:
ps aux | grep wander.py & ps aux | grep esp32_uart_logger.py
kill <PID>

// to view logs:
nano wander.log
nano esp32_logger.log

```
