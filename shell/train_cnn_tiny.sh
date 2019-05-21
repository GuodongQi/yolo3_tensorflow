NET_TYPE="cnn"
TINY=True
ANCHOR_PATH="./model_data/yolo_anchors_tiny.txt"
PRETRAIN_PATH=""

epoch=200
batch_size=4
learning_rate=1e-4

debug=False

if [ -z "${PRETRAIN_PATH}" ]

then

cmd="python train.py \
-n "${NET_TYPE}" \
-t ${TINY} \
-e ${epoch} \
-b ${batch_size} \
-lr ${learning_rate} \
-d ${debug} \
--anchor_path ${ANCHOR_PATH}
"

else

cmd="python train.py \
-n "${NET_TYPE}" \
-t ${TINY} \
-pt "${PRETRAIN_PATH}" \
-e ${epoch} \
-b ${batch_size} \
-lr ${learning_rate} \
-d ${debug} \
--anchor_path ${ANCHOR_PATH}
"

fi

echo $cmd
$cmd



