import os
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from datasets import load_dataset

model_id = ""
