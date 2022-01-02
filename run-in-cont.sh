XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run -it                                          \
    --env="DISPLAY=$DISPLAY"                            \
    --env="QT_X11_NO_MITSHM=1"                          \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"         \
    --env="XAUTHORITY=$XAUTH"                           \
    --volume="$XAUTH:$XAUTH"                            \
    --network="host"                                    \
    --pid host                                          \
    --name cont_tc                                      \
    -v /home/sampreets3/GitHub/treecounter-ML:/home/app \
    -w="/home/app/"                                     \
    --gpus all                                          \
    treecounter-ml                                      \
    python3 test.py
