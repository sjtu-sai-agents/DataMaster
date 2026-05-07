#!/bin/bash

rsync -avzP --info=progress2 -e "ssh" xiyuan@dp_gpu_4090_1:/home/xiyuan 