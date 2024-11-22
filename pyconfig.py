import yaml
class Config:
    instance = None

    def __new__(cls, path=None):
        if cls.instance is None and path is not None:
            with open(path, "r") as file:
                cls.instance = yaml.safe_load(file)
        return cls.instance

    @staticmethod
    def get_instance():
        if Config.instance is None:
            raise ValueError("Config is not initialized")
        return Config.instance