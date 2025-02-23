class Audioer:
    @staticmethod
    def deserialize(body, query=None):
        return {"inputs": bytes(body)}

    @staticmethod
    def serialize(body, accept=None):
        raise NotImplementedError("Audio serialization not implemented")
