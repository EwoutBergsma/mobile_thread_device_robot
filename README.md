# 


```cmd

ssh fieldlab@10.149.24.26


// to start the experiment:
// nohup python3 wander.py > wander.log 2>&1 &
// nohup python3 esp32_uart_logger.py > esp32_logger.log 2>&1 &

nohup python3 wander.py > wander.log 2>&1 & nohup python3 esp32_uart_logger.py > esp32_uart_logger.log 2>&1 &
nohup python3 wall_follow_and_log.py > wall_follow_and_log.log 2>&1 & nohup python3 esp32_uart_logger.py > esp32_uart_logger.log 2>&1 &

// to stop the node:
ps aux | grep wander.py & ps aux | grep esp32_uart_logger.py & ps aux | grep wall_follow_and_log.py
kill <PID>

// to view logs:
nano wander.log
nano esp32_logger.log

```



```
ros2 service call /robot_1/wall_follow/_action/cancel_goal action_msgs/srv/CancelGoal \
"{goal_info: {goal_id: {uuid: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}, stamp: {sec: 0, nanosec: 0}}}"

```
