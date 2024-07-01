#!/usr/bin/env bash

pyenv local 3.8.12
poetry env use $(which python)
poetry install
