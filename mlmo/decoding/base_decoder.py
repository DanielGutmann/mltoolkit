from mlutils.tools.signed_object import SignedObject


class BaseDecoder(SignedObject):
    
    def decode(self, **kwargs):
        raise NotImplementedError
