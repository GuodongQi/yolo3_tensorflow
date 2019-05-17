NET_TYPE="mobilenetv2"
TINY=True
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
-d ${debug}"

else

cmd="python train.py \
-n "${NET_TYPE}" \
-t ${TINY} \
-pt "${PRETRAIN_PATH}" \
-e ${epoch} \
-b ${batch_size} \
-lr ${learning_rate} \
-d ${debug}"

fi

echo $cmd
nohup stdbuf -oL $cmd > ${NET_TYPE}_${TINY}.txt &



