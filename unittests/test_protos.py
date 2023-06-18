from multiprocessing import Process

import numpy as np
import pytest


from QGrain.models import DistributionType, KernelType, SSUResult, EMMAResult, UDMResult
from QGrain.distributions import get_distribution
from QGrain.generate import random_dataset, SIMPLE_PRESET
from QGrain.protos import qgrain_pb2
from QGrain.protos.client import QGrainClient
from QGrain.protos.server import QGrainServicer


class TestServer:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=100)

    def test_get_statistics(self):
        server = QGrainServicer()
        request = qgrain_pb2.StatisticalRequest(dataset=QGrainClient._to_dataset_pb2(self.dataset))
        response = server.get_statistics(request, None)
        if len(response.results) != 0:
            results = [QGrainClient._statistical_result_pb2_to_dict(result) for result in response.results]
            assert len(results) == len(self.dataset)
        else:
            print(response.message)

    def test_get_ssu_result(self):
        server = QGrainServicer()
        sample_pb2 = QGrainClient._to_sample_pb2(self.dataset[0])
        distribution_type_pb2 = DistributionType.Normal.value
        x0_pb2 = bytes()
        settings = dict(loss="lmse", optimizer="L-BFGS-B", try_global=False, global_max_niter=100,
                        global_niter_success=5, global_step_size=0.2,
                        optimizer_max_niter=10000, need_history=True)
        request = qgrain_pb2.SSURequest(sample=sample_pb2, distribution_type=distribution_type_pb2,
                                        n_components=3, x0=x0_pb2, **settings)
        response = server.get_ssu_result(request, None)
        # if not success, the `parameters` is empty bytes
        if len(response.parameters) != 0:
            shape = (response.n_iterations, response.n_parameters, response.n_components)
            parameters = np.frombuffer(response.parameters, np.float32).copy().reshape(shape)
            result = SSUResult(self.dataset[0], DistributionType.Normal, parameters, response.time_spent)
        else:
            print(response.message)

    def test_get_emma_result(self):
        server = QGrainServicer()
        dataset_pb2 = QGrainClient._to_dataset_pb2(self.dataset)
        settings = dict(device="cpu", loss="lmse", pretrain_epochs=0, min_epochs=100,
                        max_epochs=10000, precision=6, learning_rate=5e-3, betas=(0.8, 0.5),
                        update_end_members=True, need_history=True)
        request = qgrain_pb2.EMMARequest(dataset=dataset_pb2, distribution_type=KernelType.Normal.value, n_members=3,
                                         x0=bytes(), **settings)
        response = server.get_emma_result(request, None)
        if len(response.losses) != 0:
            proportions_shape = (response.n_iterations, response.n_samples, response.n_members)
            end_members_shape = (response.n_iterations, response.n_members, response.n_classes)
            proportions = np.frombuffer(response.proportions, dtype=np.float32).copy().reshape(proportions_shape)
            end_members = np.frombuffer(response.end_members, dtype=np.float32).copy().reshape(end_members_shape)
            losses = {"lmse": np.array(response.losses, dtype=np.float32)}
            result = EMMAResult(self.dataset, KernelType.Normal, 3, proportions, end_members, response.time_spent,
                                x0=None, loss_series=losses, settings=settings)
        else:
            print(response.message)

    def test_get_udm_result(self):
        server = QGrainServicer()
        dataset_pb2 = QGrainClient._to_dataset_pb2(self.dataset)
        n_parameters = get_distribution(DistributionType.Normal).N_PARAMETERS + 1
        settings = dict(device="cpu", pretrain_epochs=0, min_epochs=100, max_epochs=10000, precision=6,
                        learning_rate=5e-3, betas=(0.8, 0.5), constraint_level=2.0, need_history=True)
        request = qgrain_pb2.UDMRequest(dataset=dataset_pb2, distribution_type=KernelType.Normal.value,
                                        n_components=3, x0=bytes(), **settings)
        response = server.get_udm_result(request, None)
        if len(response.parameters) != 0:
            shape = (response.n_iterations, response.n_samples, n_parameters, response.n_components)
            parameters = np.frombuffer(response.parameters, dtype=np.float32).copy().reshape(shape)
            losses = {"total": np.array(response.total_losses, dtype=np.float32),
                      "distribution": np.array(response.distribution_losses, dtype=np.float32),
                      "component": np.array(response.component_losses, dtype=np.float32)}
            result = UDMResult(self.dataset, KernelType.Normal, 3, parameters, response.time_spent, x0=None,
                               loss_series=losses, settings=settings)
        else:
            print(response.message)


def start_server():
    server = QGrainServicer()
    server.serve()


class TestClient:
    dataset = random_dataset(**SIMPLE_PRESET, n_samples=200)

    def setup_class(self):
        self.service_process = Process(target=start_server)
        self.service_process.start()

    def teardown_class(self):
        self.service_process.terminate()

    def test_get_statistics(self):
        client = QGrainClient()
        results = client.get_statistics(self.dataset)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)

    def test_get_ssu_result(self):
        client = QGrainClient()
        for sample in self.dataset:
            result = client.get_ssu_result(sample, DistributionType.Normal, 3)
            assert isinstance(result, SSUResult)

    def test_get_emma_result(self):
        client = QGrainClient()
        result = client.get_emma_result(self.dataset, KernelType.Normal, 3)
        assert isinstance(result, EMMAResult)

    def test_get_udm_result(self):
        client = QGrainClient()
        result = client.get_udm_result(self.dataset, KernelType.Normal, 3)
        assert isinstance(result, UDMResult)


if __name__ == "__main__":
    pytest.main(["-s"])
