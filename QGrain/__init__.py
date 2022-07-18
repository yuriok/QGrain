import os

QGRAIN_VERSION = "0.5.0.0"
QGRAIN_ROOT_PATH = os.path.dirname(__file__)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="QGrain is an easy-to-use software for the analysis of grain size distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--server", action="store_true", default=False,
                        help="start a local grpc server to handle the requests of computation")
    parser.add_argument("--address", type=str, default="localhost:50051",
                        help="specify the local ip address used to start the grpc server")
    parser.add_argument("--max-workers", type=int, default=4, help="specify the max number of thread workers.",
                        dest="max_workers")
    parser.add_argument("--max-message-length", type=int, default=2**30,
                        help="specify the max length of grpc messages (bytes).", dest="max_message_length")
    parser.add_argument("--max-dataset-size", type=int, default=100000, help="specify the max size of dataset.",
                        dest="max_dataset_size")
    parser.add_argument("--target", type=str, default="localhost:50051",
                        help="specify the remote ip address of the grpc server")
    args = parser.parse_args()
    if args.server:
        from .protos.server import QGrainServicer
        qgrain_server = QGrainServicer(
            address=args.address, max_workers=args.max_workers, max_message_length=args.max_message_length,
            max_dataset_size=args.max_dataset_size)
        qgrain_server.serve()
    else:
        from .protos.client import QGrainClient
        from .ui.MainWindow import qgrain_app
        QGrainClient.set_target(args.target)
        qgrain_app()
