__all__ = ['Initializer']

import logging
class Initializer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        self.logger.info("Initializing...")
        self.logger.info("Initializing done.")
        return self.config