# Imports de bibliothèques tierces
import torch
import yaml

# Imports locaux
import script_pytorch.configuration as configuration

configuration.update_device_in_config("config.yaml")