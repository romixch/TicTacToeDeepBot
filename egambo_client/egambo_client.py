from .tcpjson import TcpJson


class EgamboClient:

    def __init__(self, width = 3, height = 3, runlength = 3):
        self.host = "52.53.136.213"
        self.port = 8559


    def __enter__(self):
        self.tcpjson = TcpJson(self.host, self.port, True)


    def __exit__(self, type, value, traceback):
        self.tcpjson.close()


    def login(self, user, password):
        self.tcpjson.Send('wuhuu')