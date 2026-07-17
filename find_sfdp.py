import os

search_dir = r"E:\Anaconda\envs\py-dev"
found = []

for root, dirs, files in os.walk(search_dir):
    for f in files:
        if f.lower() in ("sfdp.exe", "sfdp.bat"):
            path = os.path.join(root, f)
            found.append(path)
            print(f"Found: {path}")

if not found:
    print("sfdp executable not found in E:\\Anaconda\\envs\\py-dev")
