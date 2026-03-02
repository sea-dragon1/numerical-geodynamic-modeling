import paramiko
import subprocess

def pack_folder_on_remote(hostname, port, username, password, remote_path,remote_archive_path):
    """
    使用rsync从远程服务器下载文件到本地
    :param hostname: 远程服务器的主机名或IP地址
    :param port: 远程服务器的SSH端口
    :param username: 登录远程服务器的用户名
    :param password: 登录远程服务器的密码
    :param remote_path: 远程服务器上的文件或文件夹路径
    :param local_path: 本地保存文件或文件夹的路径
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=username, password=password)

# 构建打包命令，这里使用tar命令进行打包
        pack_command = f"tar -czf {remote_archive_path} {remote_path}"

        # 执行打包命令
        stdin, stdout, stderr = ssh.exec_command(pack_command)

        # 读取命令执行结果
        result = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if error:
            print(f"打包失败，错误信息: {error}")
        else:
            print(f"文件夹 {remote_path} 已成功打包为 {remote_archive_path}")

        # # 构建rsync命令
        # rsync_command = f"rsync -avz -e'ssh -p {port}' {username}@{hostname}:{remote_path} {local_path}"

        # # 执行rsync命令
        # result = subprocess.run(rsync_command, shell=True, check=True, capture_output=True, text=True)

        # if result.returncode == 0:
        #     print("文件下载成功")
        # else:
        #     print(f"文件下载失败，错误信息: {result.stderr}")

        ssh.close()
    except paramiko.AuthenticationException:
        print("认证失败，请检查用户名和密码")
    except paramiko.SSHException as e:
        print(f"SSH连接错误: {e}")
    except subprocess.CalledProcessError as e:
        print(f"rsync命令执行错误: {e}")
    except Exception as e:
        print(f"发生其他错误: {e}")

def download_file_from_remote(hostname, port, username, password, remote_file_path, local_file_path):
    """
    从远程服务器下载文件到本地
    :param hostname: 远程服务器的主机名或IP地址
    :param port: 远程服务器的SSH端口
    :param username: 登录远程服务器的用户名
    :param password: 登录远程服务器的密码
    :param remote_file_path: 远程服务器上的文件路径
    :param local_file_path: 本地保存文件的路径
    """
    try:
        transport = paramiko.Transport((hostname, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        sftp.get(remote_file_path, local_file_path)
        print(f"文件 {remote_file_path} 已成功下载到 {local_file_path}")

        sftp.close()
        transport.close()
    except paramiko.AuthenticationException:
        print("认证失败，请检查用户名和密码")
    except paramiko.SSHException as e:
        print(f"SSH连接错误: {e}")
    except FileNotFoundError:
        print(f"远程文件 {remote_file_path} 未找到")
    except Exception as e:
        print(f"发生其他错误: {e}")

# 示例调用
if __name__ == "__main__":

    port = 22  # SSH端口，默认是22
    hostname = "192.168.10.51"
    username = "maththu57"
    password = r"nMy%AsDYqfBK"
    

    remote_path = "/fs2/home/maththu57/lhl/test0409/t8/output_t8/solution" # 远程服务器上的文件夹路径
    remote_archive_path = "/fs2/home/maththu57/lhl/test0409/t8/output_t8/solution.tar.gz"  # 远程服务器上打包后的文件路径
    # local_path = r"E:\backup\DoubleSubduction\\model\double_plstc_subeen\\gwb_add\\ts\\t8\\output_t8\solution"  # 本地保存文件的文件夹路径
    local_path = r"E:\backup\DoubleSubduction\\model\double_plstc_subeen\\gwb_add\\ts\\t8\\output_t8"  # 本地保存文件的文件夹路径

    pack_folder_on_remote(hostname, port, username, password, remote_path,remote_archive_path)
    download_file_from_remote(hostname, port, username, password, remote_archive_path, local_path)  # 下载打包后的文件