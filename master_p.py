import os, socket
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
# Specify number of GPU slot per node below
print(IPAddr + " slots=1")