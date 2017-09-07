import unittest
import numpy
from egambo_client.egambo_client import EgamboClient

class EgamboClientTests(unittest.TestCase):

    def test_connect_and_disconnect(self):
        with EgamboClient() as client:
            print(client)


    def test_login(self):
        with EgamboClient() as client:
            client.login('roman', 'qwerty123456')
