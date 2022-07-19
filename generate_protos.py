import os

proto_dir = "./QGrain/protos"
target_dir = "./QGrain/protos"

if __name__ == "__main__":
    for filename in os.listdir(proto_dir):
        if filename[-6:] == ".proto":
            command = f"python -m grpc_tools.protoc -I {proto_dir} --python_out={target_dir} --grpc_python_out={target_dir} {filename}"
            print(command)
            os.system(command)
