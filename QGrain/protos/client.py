from typing import *

import grpc
import numpy as np
from numpy import ndarray

from ..models import DistributionType, KernelType, Sample, SSUResult, Dataset, ArtificialDataset, EMMAResult, UDMResult
from ..distributions import get_distribution
from . import qgrain_pb2 as qgrain_pb2
from . import qgrain_pb2_grpc as qgrain_pb2_grpc

MAX_MESSAGE_LENGTH = 2**30


class QGrainClient:
    _target = ""

    def __init__(self):
        pass

    @classmethod
    def set_target(cls, target: str):
        cls._target = target

    @classmethod
    def _to_sample_pb2(cls, sample: Union[ArtificialDataset, Sample]) -> qgrain_pb2.Sample:
        return qgrain_pb2.Sample(name=sample.name, classes=sample.classes, distribution=sample.distribution)

    @classmethod
    def _to_dataset_pb2(cls, dataset: Union[ArtificialDataset, Dataset]) -> qgrain_pb2.Dataset:
        dataset_pb2 = qgrain_pb2.Dataset(name=dataset.name, n_samples=len(dataset), n_classes=len(dataset.classes),
                                         sample_names=dataset.sample_names, classes=dataset.classes,
                                         distributions=dataset.distributions.astype(np.float32).tobytes())
        return dataset_pb2

    @classmethod
    def _statistical_result_pb2_to_dict(cls, response: qgrain_pb2.StatisticalRequest) -> Dict:
        arithmetic_keys = ("mean", "std", "skewness", "kurtosis")
        methods = ("geometric", "logarithmic", "geometric_fw57", "logarithmic_fw57")
        method_keys = ("mean", "std", "skewness", "kurtosis", "median", "mode", "modes",
                       "mean_description", "std_description", "skewness_description", "kurtosis_description")
        other_keys = ("proportions_gsm", "proportions_ssc", "proportions_bgssc", "proportions",
                      "group_folk54", "group_bp12_symbol", "group_bp12")
        statistics = {"arithmetic": {key: getattr(response.arithmetic, key) for key in arithmetic_keys}}
        for method in methods:
            statistics[method] = {key: getattr(getattr(response, method), key) for key in method_keys}
        for key in other_keys:
            statistics[key] = getattr(response, key)
        return statistics

    @property
    def has_target(self) -> bool:
        return self._target != ""

    def get_service_state(self):
        with grpc.insecure_channel(self._target, options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)]) as channel:
            stub = qgrain_pb2_grpc.QGrainStub(channel)
            request = qgrain_pb2.ServiceStateRequest()
            response = stub.get_service_state(request)
            state = dict(max_workers=response.max_workers, max_message_length=response.max_message_length,
                         available_devices=tuple(response.available_devices),
                         max_dataset_size=response.max_dataset_size)
            return state

    def get_statistics(self, dataset: Union[ArtificialDataset, Dataset]) -> Optional[List[Dict[str, Any]]]:
        with grpc.insecure_channel(self._target, options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = qgrain_pb2_grpc.QGrainStub(channel)
            request = qgrain_pb2.StatisticalRequest(dataset=self._to_dataset_pb2(dataset))
            response = stub.get_statistics(request)
            if len(response.results) != 0:
                results = [self._statistical_result_pb2_to_dict(result) for result in response.results]
                return results
            else:
                return response.message

    def get_ssu_result(self, sample: Union[ArtificialDataset, Sample], distribution_type: DistributionType,
                       n_components: int, x0: ndarray = None, loss: str = "lmse", optimizer: str = "L-BFGS-B",
                       try_global: bool = False, global_max_niter: int = 100, global_niter_success: int = 5,
                       global_step_size: float = 0.2, optimizer_max_niter: int = 10000, need_history: bool = True) -> \
            Union[SSUResult, str]:
        with grpc.insecure_channel(self._target, options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = qgrain_pb2_grpc.QGrainStub(channel)
            sample_pb2 = self._to_sample_pb2(sample)
            distribution_type_pb2 = distribution_type.value
            x0_pb2 = bytes() if x0 is None else x0.astype(np.float32).tobytes()
            settings = dict(loss=loss, optimizer=optimizer, try_global=try_global, global_max_niter=global_max_niter,
                            global_niter_success=global_niter_success, global_step_size=global_step_size,
                            optimizer_max_niter=optimizer_max_niter, need_history=need_history)
            request = qgrain_pb2.SSURequest(sample=sample_pb2, distribution_type=distribution_type_pb2,
                                            n_components=n_components, x0=x0_pb2, **settings)
            response = stub.get_ssu_result(request)
            # if not success, the `parameters` is empty bytes
            if len(response.parameters) != 0:
                shape = (response.n_iterations, response.n_parameters, response.n_components)
                parameters = np.frombuffer(response.parameters, np.float32).copy().reshape(shape)
                result = SSUResult(sample, distribution_type, parameters, response.time_spent, x0=x0, settings=settings)
                return result
            else:
                return response.message

    def get_emma_result(self, dataset: Union[ArtificialDataset, Dataset], kernel_type: KernelType, n_members: int,
                        x0: ndarray = None, device="cpu", loss="lmse", pretrain_epochs=0, min_epochs=100,
                        max_epochs=10000, precision: Union[int, float] = 6, learning_rate=5e-3, betas=(0.8, 0.5),
                        update_end_members=True, need_history=True) -> Union[EMMAResult, str]:
        with grpc.insecure_channel(self._target, options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = qgrain_pb2_grpc.QGrainStub(channel)
            dataset_pb2 = self._to_dataset_pb2(dataset)
            x0_pb2 = bytes() if x0 is None else x0.astype(np.float32).tobytes()
            settings = dict(device=device, loss=loss, pretrain_epochs=pretrain_epochs, min_epochs=min_epochs,
                            max_epochs=max_epochs, precision=precision, learning_rate=learning_rate, betas=betas,
                            update_end_members=update_end_members, need_history=need_history)
            request = qgrain_pb2.EMMARequest(dataset=dataset_pb2, distribution_type=kernel_type.value,
                                             n_members=n_members, x0=x0_pb2, **settings)
            response = stub.get_emma_result(request)
            if len(response.losses) != 0:
                proportions_shape = (response.n_iterations, response.n_samples, response.n_members)
                end_members_shape = (response.n_iterations, response.n_members, response.n_classes)
                proportions = np.frombuffer(response.proportions, dtype=np.float32).copy().reshape(proportions_shape)
                end_members = np.frombuffer(response.end_members, dtype=np.float32).copy().reshape(end_members_shape)
                losses = {loss: np.array(response.losses, dtype=np.float32)}
                result = EMMAResult(dataset, kernel_type, n_members, proportions, end_members, response.time_spent,
                                    x0=x0, loss_series=losses, settings=settings)
                return result
            else:
                return response.message

    def get_udm_result(self, dataset: Union[ArtificialDataset, Dataset], kernel_type: KernelType, n_components: int,
                       x0: ndarray = None, device="cpu", pretrain_epochs=0, min_epochs=100, max_epochs=10000,
                       precision: Union[int, float] = 6, learning_rate=5e-3, betas=(0.8, 0.5),
                       constraint_level: Union[int, float] = 2.0, need_history=True) -> Union[UDMResult, str]:
        with grpc.insecure_channel(self._target, options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
            stub = qgrain_pb2_grpc.QGrainStub(channel)
            dataset_pb2 = self._to_dataset_pb2(dataset)
            x0_pb2 = bytes() if x0 is None else x0.astype(np.float32).tobytes()
            n_parameters = get_distribution(DistributionType.__members__[kernel_type.name]).N_PARAMETERS + 1
            settings = dict(device=device, pretrain_epochs=pretrain_epochs, min_epochs=min_epochs,
                            max_epochs=max_epochs, precision=precision, learning_rate=learning_rate, betas=betas,
                            constraint_level=constraint_level, need_history=need_history)
            request = qgrain_pb2.UDMRequest(dataset=dataset_pb2, distribution_type=kernel_type.value,
                                            n_components=n_components, x0=x0_pb2, **settings)
            response = stub.get_udm_result(request)
            if len(response.parameters) != 0:
                shape = (response.n_iterations, response.n_samples, n_parameters, response.n_components)
                parameters = np.frombuffer(response.parameters, dtype=np.float32).copy().reshape(shape)
                losses = {"total": np.array(response.total_losses, dtype=np.float32),
                          "distribution": np.array(response.distribution_losses, dtype=np.float32),
                          "component": np.array(response.component_losses, dtype=np.float32)}
                result = UDMResult(dataset, kernel_type, n_components, parameters, response.time_spent, x0=x0,
                                   loss_series=losses, settings=settings)
                return result
            else:
                return response.message
