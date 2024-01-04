import os, socket, time
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

# Open a TCP connection to the master.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((os.environ["CDSW_MASTER_IP"], 6000))
s.send(IPAddr.encode())
data = s.recv(1024)
s.close()

while True:
    try:
        # Add any tasks or commands you want to execute here.
        pass
    except Exception as e:
        print("Error:", str(e))