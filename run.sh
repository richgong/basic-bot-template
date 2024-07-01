#!/usr/bin/env bash

if [ ! -f .env ]; then
    echo "[ERROR] Please put your OpenAI environment variables in .env"
    exit 1
fi


poetry run dotenv run python example_bot.py
