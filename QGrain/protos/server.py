import logging
import string
from concurrent import futures

import grpc
import numpy as np

from ..statistics import to_phi, all_statistics
from ..models import DistributionType, KernelType, Sample, SSUResult, validate_distributions, validate_classes, Dataset
from ..distributions import get_distribution
from ..ssu import try_ssu
from ..emma import try_emma
from ..udm import try_udm
from . import qgrain_pb2 as qgrain_pb2
from . import qgrain_pb2_grpc as qgrain_pb2_grpc


class QGrainServicer(qgrain_pb2_grpc.QGrainServicer):
    logger = logging.getLogger("QGrain.QGrainServicer")

    def __init__(self, address: str = "localhost:50051", max_workers: int = 8, max_message_length: int = 2**30,
                 max_dataset_size: int = 100000):
        super(QGrainServicer, self).__init__()
        self._address = address
        self._max_workers = max_workers
        self._max_message_length = max_message_length
        self._max_dataset_size = max_dataset_size
    
    @classmethod
    def _to_sample(cls, sample_pb2) -> Sample:
        if len(sample_pb2.classes) == 0:
            raise ValueError("The series of grain size classes is empty.")
        if len(sample_pb2.distribution) == 0:
            raise ValueError("The series of grain size distribution is empty.")
        if len(sample_pb2.classes) != len(sample_pb2.distribution):
            raise ValueError("The lengths of grain size classes and frequencies are not equal.")
        valid, array_or_msg = validate_classes(sample_pb2.classes)
        if not valid:
            raise ValueError(array_or_msg)
        classes = array_or_msg.astype(np.float32)
        classes_phi = to_phi(classes)
        valid, array_or_msg = validate_distributions([sample_pb2.distribution])
        if not valid:
            raise ValueError(array_or_msg)
        distribution = array_or_msg.astype(np.float32)[0]
        sample = Sample(sample_pb2.name, classes, classes_phi, distribution)
        return sample

    @classmethod
    def _to_dataset(cls, dataset_pb2: qgrain_pb2.Dataset) -> Dataset:
        sample_names = list(dataset_pb2.sample_names)
        classes = np.array(dataset_pb2.classes, dtype=np.float32)
        shape = (dataset_pb2.n_samples, dataset_pb2.n_classes)
        distributions = np.frombuffer(dataset_pb2.distributions, dtype=np.float32).copy().reshape(shape)
        dataset = Dataset(dataset_pb2.name, sample_names, classes, distributions)
        return dataset

    def get_service_state(self, request: qgrain_pb2.ServiceStateRequest, context):
        devices = ["cpu"]
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        if "cuda:0" in devices:
            devices.remove("cuda:0")
        state = dict(max_workers=self._max_workers, max_message_length=self._max_message_length,
                     max_dataset_size=self._max_dataset_size, available_devices=tuple(devices))
        response = qgrain_pb2.ServiceStateResponse(**state)
        return response

    def get_statistics(self, request: qgrain_pb2.StatisticalRequest, context):
        try:
            dataset = self._to_dataset(request.dataset)
        except TypeError as e:
            return qgrain_pb2.StatisticalResponse(message=f"The dataset in request is invalid, please check: {e}.")
        except ValueError as e:
            return qgrain_pb2.StatisticalResponse(message=f"The dataset in request is invalid, please check: {e}.")
        results = []
        for sample in dataset:
            all_parameters = all_statistics(sample.classes, sample.classes_phi, sample.distribution)
            methods = ("arithmetic", "geometric", "logarithmic", "geometric_fw57", "logarithmic_fw57")
            parameters_of_methods = {}
            for method in methods:
                parameters = qgrain_pb2.StatisticalParameters(method=method, **all_parameters[method])
                parameters_of_methods[method] = parameters
            other_keys = ("proportions_gsm", "proportions_ssc", "proportions_bgssc",
                          "group_folk54", "group_bp12_symbol", "group_bp12")
            other_parameters = {key: all_parameters[key] for key in other_keys}
            proportions = {string.capwords(f"{adj} {grade}"): proportion for (adj, grade), proportion in
                           all_parameters["proportions"].items()}
            result = qgrain_pb2.StatisticalResult(**parameters_of_methods, **other_parameters, proportions=proportions)
            results.append(result)
        message = f"Success! {len(dataset)} statistical results of dataset {dataset.name} are available."
        response = qgrain_pb2.StatisticalResponse(message=message, results=results)
        return response

    def get_ssu_result(self, request: qgrain_pb2.SSURequest, context):
        try:
            sample = self._to_sample(request.sample)
        except ValueError as e:
            return qgrain_pb2.SSUResponse(message=f"The sample in request is invalid, please check: {e}.")
        distribution_type = {t.value: t for name, t in DistributionType.__members__.items()}[request.distribution_type]
        distribution_class = get_distribution(distribution_type)
        n_parameters = distribution_class.N_PARAMETERS + 1
        if len(request.x0) == 0:
            x0 = None
        else:
            x0 = np.frombuffer(request.x0, dtype=np.float32).copy().reshape(n_parameters, request.n_components)
        settings = dict(loss=request.loss, optimizer=request.optimizer, try_global=request.try_global,
                        global_max_niter=request.global_max_niter, global_niter_success=request.global_niter_success,
                        global_step_size=request.global_step_size, optimizer_max_niter=request.optimizer_max_niter,
                        need_history=request.need_history)
        try:
            result, message = try_ssu(sample, distribution_type, request.n_components, x0=x0,
                                      **settings, logger=self.logger)
        except AssertionError as e:
            return qgrain_pb2.SSUResponse(message=f"The algorithm settings in request is invalid, please check: {e}.")
        if isinstance(result, SSUResult):
            response = qgrain_pb2.SSUResponse(
                message=f"Success! The SSU result of sample {sample.name} is available.", time_spent=result.time_spent,
                n_iterations=result.n_iterations, n_parameters=result.n_parameters, n_components=len(result),
                parameters=result.parameters.astype(np.float32).tobytes())
            return response
        else:
            response = qgrain_pb2.SSUResponse(message=message)
            return response

    def get_emma_result(self, request: qgrain_pb2.EMMARequest, context):
        try:
            dataset = self._to_dataset(request.dataset)
        except TypeError as e:
            return qgrain_pb2.EMMAResponse(message=f"The dataset in request is invalid, please check: {e}.")
        except ValueError as e:
            return qgrain_pb2.EMMAResponse(message=f"The dataset in request is invalid, please check: {e}.")
        distribution_type = {t.value: t for name, t in DistributionType.__members__.items()}[request.distribution_type]
        kernel_type = {t.value: t for name, t in KernelType.__members__.items()}[request.distribution_type]
        n_parameters = get_distribution(distribution_type).N_PARAMETERS
        if len(request.x0) == 0:
            x0 = None
        else:
            x0 = np.frombuffer(request.x0, dtype=np.float32).copy().reshape(n_parameters, request.n_members)
        settings = dict(device=request.device, loss=request.loss, pretrain_epochs=request.pretrain_epochs,
                        min_epochs=request.min_epochs, max_epochs=request.max_epochs, precision=request.precision,
                        learning_rate=request.learning_rate, betas=tuple(request.betas),
                        update_end_members=request.update_end_members, need_history=request.need_history)
        try:
            result = try_emma(dataset, kernel_type, request.n_members, x0=x0, **settings, logger=self.logger)
        except AssertionError as e:
            return qgrain_pb2.EMMAResponse(message=f"The algorithm settings in request is invalid, please check: {e}.")
        response = qgrain_pb2.EMMAResponse(
            message=f"Success! The EMMA result of dataset {dataset.name} is available.", time_spent=result.time_spent,
            n_iterations=result.n_iterations, n_samples=result.n_samples, n_members=result.n_members,
            n_classes=result.n_classes, proportions=result._proportions.astype(np.float32).tobytes(),
            end_members=result._end_members.astype(np.float32).tobytes(), losses=result.loss_series(request.loss))
        return response

    def get_udm_result(self, request: qgrain_pb2.UDMRequest, context):
        try:
            dataset = self._to_dataset(request.dataset)
        except TypeError as e:
            return qgrain_pb2.UDMResponse(message=f"The dataset in request is invalid, please check: {e}.")
        except ValueError as e:
            return qgrain_pb2.UDMResponse(message=f"The dataset in request is invalid, please check: {e}.")
        distribution_type = {t.value: t for name, t in DistributionType.__members__.items()}[request.distribution_type]
        kernel_type = {t.value: t for name, t in KernelType.__members__.items()}[request.distribution_type]
        n_parameters = get_distribution(distribution_type).N_PARAMETERS
        if len(request.x0) == 0:
            x0 = None
        else:
            x0 = np.frombuffer(request.x0, dtype=np.float32).copy().reshape(n_parameters, request.n_components)
        settings = dict(device=request.device, pretrain_epochs=request.pretrain_epochs, min_epochs=request.min_epochs,
                        max_epochs=request.max_epochs, precision=request.precision, learning_rate=request.learning_rate,
                        betas=tuple(request.betas), constraint_level=request.constraint_level,
                        need_history=request.need_history)
        try:
            result = try_udm(dataset, kernel_type, request.n_components, x0=x0, **settings, logger=self.logger)
        except AssertionError as e:
            return qgrain_pb2.UDMResponse(
                message=f"The algorithm settings in request is invalid, please check: {e}.")
        response = qgrain_pb2.UDMResponse(
            message=f"Success! The UDM result of dataset {dataset.name} is available.", time_spent=result.time_spent,
            n_iterations=result.n_iterations, n_samples=result.n_samples, n_components=result.n_components,
            n_classes=result.n_classes, parameters=result.parameters.astype(np.float32).tobytes(),
            distribution_losses=result.loss_series("distribution"), component_losses=result.loss_series("component"),
            total_losses=result.loss_series("total"))
        return response

    def serve(self):
        self.logger.info(f"Starting the server and listening the address: {self._address}. "
                         f"Max thread workers is {self._max_workers}. "
                         f"Max length of the message is {self._max_message_length} bytes. "
                         f"Max size of the grain size dataset is {self._max_dataset_size} samples.\n")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers),
                             options=[("grpc.max_send_message_length", self._max_message_length),
                                      ("grpc.max_receive_message_length", self._max_message_length)])
        qgrain_pb2_grpc.add_QGrainServicer_to_server(QGrainServicer(), server)
        server.add_insecure_port(self._address)
        server.start()
        server.wait_for_termination()
