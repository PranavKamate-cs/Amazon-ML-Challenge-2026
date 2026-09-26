import paramiko
import sys

def run(cmd):
    key_path = r'C:\Users\Pranav\Downloads\ml-challenge-key.pem'
    ip = '3.88.2.66'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname=ip, username='ubuntu', key_filename=key_path, timeout=10)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        print("=== STDOUT ===")
        print(out.encode('ascii', errors='replace').decode('ascii'))
        if err:
            print("=== STDERR ===")
            print(err.encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cat run_preprocess.log; ls -lh data_preprocessed/test/ data_preprocessed/train/ 2>/dev/null; ps aux | grep [p]ython3"
    run(cmd)
