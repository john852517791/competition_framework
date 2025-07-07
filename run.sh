# nohup python main.py --config a_logs/efficientnet-b4/version_2/config.yaml &
# nohup python main.py --config a_logs/efficientnet-b5/version_2/config.yaml &
# nohup python main.py --config a_logs/efficientnet-b6/version_2/config.yaml &
# nohup python main.py --config a_logs/efficientnet-b7/version_2/config.yaml &
# nohup python main.py --config a_logs/efficientnet-b8/version_2/config.yaml &
# nohup python main.py --config a_logs/efficientnet-l2/version_2/config.yaml &

#!/bin/bash

echo "Using seq:"
for i in $(seq 0 6); do
  eval "nohup python utils/offline_augmentation.py $i > $i.log &"
done