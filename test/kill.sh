#!/bin/bash

PORT=8890
kill -9 $(lsof -ti :$PORT)