#!/bin/bash
poetry lock && git add poetry.lock && git commit -m "Update poetry.lock remotely" && git push
