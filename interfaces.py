import abc
from abc import ABC

from pandas import DataFrame


class PipelineStep(ABC):

    def __init__(self):
        self.data = None

    @abc.abstractmethod
    def run(self, data: DataFrame) -> DataFrame:
        pass


class Pipeline(PipelineStep):

    def __init__(self):
        self.steps = []
        self.last_run = None
        pass

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def run(self, data: DataFrame) -> DataFrame:
        self.last_run = []
        self.last_run.append(data)
        previous_data = data
        for step in self.steps:
            next_data = step.run(previous_data)
            previous_data = next_data
            self.last_run.append(next_data)
        return next_data
