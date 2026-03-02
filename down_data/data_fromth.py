import paramiko
import os
import time
import re
import logging
logging.basicConfig(level=logging.DEBUG)

def paramiko_download_files(remote_files, local_files, hostname, username, password):
    """
    使用 Paramiko 批量下载文件。
    :param remote_files: 远程文件路径列表
    :param local_files: 本地文件路径列表
    :param hostname: 远程服务器的主机名或 IP 地址
    :param username: 用户名
    :param password: 密码
    """
    # 创建 SSH 客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 连接到远程服务器
        ssh.connect(hostname, username=username, password=password)
        
        # 打开 SFTP 会话
        sftp = ssh.open_sftp()
        
        # 批量下载文件
        for remote_path, local_path in zip(remote_files, local_files):
            print(f"Downloading {remote_path} to {local_path}...")
            sftp.get(remote_path, local_path)
            print(f"Downloaded {remote_path} to {local_path}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭 SFTP 会话和 SSH 连接
        sftp.close()
        ssh.close()

def paramiko_download_folders(remote_folders, local_folders, hostname, username, password, rename_existing=False, skip_existing=True):
    """
    使用 Paramiko 批量下载多个远程文件夹中的所有文件。
    :param remote_folders: 远程文件夹路径列表
    :param local_folders: 本地文件夹路径列表
    :param hostname: 远程服务器的主机名或 IP 地址
    :param username: 用户名
    :param password: 密码
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None  # 初始化 sftp 为 None

    try:
        ssh.connect(hostname, username=username, password=password)
        sftp = ssh.open_sftp()
        print("Connected to the server.")
        def download_dir(remote_dir, local_dir):
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            for item in sftp.listdir_attr(remote_dir):
                print(f"Item: {item.filename}, Mode: {oct(item.st_mode)}")
                remote_item_path = os.path.join(remote_dir, item.filename).replace("\\", "/")
                local_item_path = os.path.join(local_dir, item.filename).replace("\\", "/")
                print(f"Processing {remote_item_path}...")
                # 如果是文件夹，递归下载
                if item.st_mode & 0o40000:  # 如果是文件夹
                    download_dir(remote_item_path, local_item_path)
                else:  # 如果是文件
                    if os.path.exists(local_item_path):
                        if skip_existing:
                            print(f"Skipping existing file: {local_item_path}")
                            continue
                        elif rename_existing:
                            new_local_item_path = os.path.join(local_dir, f"{item.filename}.bak").replace("\\", "/")
                            os.rename(local_item_path, new_local_item_path)
                            print(f"Renamed existing file to: {new_local_item_path}")
                    print(f"Downloading {remote_item_path} to {local_item_path}...")
                    sftp.get(remote_item_path, local_item_path)
                    print(f"Downloaded {remote_item_path} to {local_item_path}.")  

        for remote_folder, local_folder in zip(remote_folders, local_folders):
            print(f"Downloading contents of {remote_folder} to {local_folder}...")
            download_dir(remote_folder, local_folder)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sftp:
            sftp.close()
        ssh.close()

if __name__ == "__main__":
    hostname = "192.168.10.51"
    username = "maththu57"
    password = r"nMy%AsDYqfBK"

    fileorfolder = "folder"  # "file" 或 "folder"

    remote_files_path = "/fs2/home/maththu57/lhl/test0409/t10/output_t10/solution/solution-00000.pvtu"
    local_files_path = r"E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t10\output_t10\solution\solution-00000.pvtu"


    # folder_path = '/fs2/home/maththu57/lhl/test0409/t10/output_t10/solution/'
    # pattern = r'solution-\d+\.pvtu'
    # matching_files = []

    # # 遍历文件夹下的所有文件
    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if re.match(pattern, file):
    #             file_path = os.path.join(root, file)
    #             matching_files.append(file_path)
                
    # 远程文件路径列表
    remote_files = [
        "/fs2/home/maththu57/lhl/test0409/t10/output_t10/solution/solution-00000.pvtu",
        "/fs2/home/maththu57/lhl/test0409/t10/output_t10/solution/solution-00050.pvtu"
    ]
    
    # 本地文件路径列表
    local_files = [
        r"E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t10\output_t10\solution\solution-00000.pvtu",
        r"E:\backup\DoubleSubduction\model\double_plstc_subeen\gwb_add\ts\t10\output_t10\solution\solution-00050.pvtu"
    ]

    remote_folders = [
            # "/fs2/home/maththu57/lhl/test0409/t9/output_t9/solution",
            "/fs2/home/maththu57/lhl/test0409/t12/output_t11/solution",
        ]
        
        # 本地文件夹路径列表
    local_folders = [
            # r"E:\backup\DoubleSubduction\\model\double_plstc_subeen\\gwb_add\\ts\\t9\\output_t9\solution",
            "E:/backup/DoubleSubduction/model/double_plstc_subeen/gwb_add/ts/t12/output_t11/solution",
        ]
         
    # 调用函数批量下载文件
    # paramiko_download_files(remote_files, local_files, hostname, username, password)

    if fileorfolder == "folder":
        # 远程文件夹路径列表
        print("Downloading folders...")
        # 调用函数批量下载文件夹
        paramiko_download_folders(remote_folders, local_folders, hostname, username, password, False, skip_existing=True)
    
    elif fileorfolder == "file":
        paramiko_download_files(remote_files, local_files, hostname, username, password)