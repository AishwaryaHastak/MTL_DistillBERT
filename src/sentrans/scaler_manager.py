# scaler_manager.py
from sklearn.preprocessing import StandardScaler

class ScalerManager:
    _instance = None

    @staticmethod
    def get_instance():
        if ScalerManager._instance is None:
            ScalerManager()
        return ScalerManager._instance

    def __init__(self):
        if ScalerManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ScalerManager._instance = self
            self.scaler = StandardScaler()

    def get_scaler(self):
        return self.scaler
