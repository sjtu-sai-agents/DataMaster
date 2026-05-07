# -L 参数：将远程服务器的端口映射到本地
autossh -M 0 -N -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 7777:localhost:7777 root@Comp_cpu_1 -vv
autossh -M 0 -N -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8899:localhost:8899 root@Comp_cpu_1 -vv

# -R 参数：将本地端口映射到服务器
autossh -M 0 -N -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -R 8899:localhost:8899 root@Comp_cpu_1 -vv


autossh -M 0 -N -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -R 9903:localhost:9903 root@Comp_gpu_3 -vv
