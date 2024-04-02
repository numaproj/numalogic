class UnRecognizedAWSClientException(Exception):

    # Constructor or Initializer
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ConfigNotFoundError(Exception):

    # Constructor or Initializer
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class UnRecognizedDatabaseTypeException(Exception):

    # Constructor or Initializer
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class UnRecognizedDatabaseServiceProviderException(Exception):

    # Constructor or Initializer
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)