class ReflectionStrategy:
    """
    Base class for reflection behaviors.
    """

    def reflect(self, agent):
        """
        Perform a reflection step. 
        Receives the agent so it can modify Q, epsilon, memory, etc.
        """
        raise NotImplementedError
