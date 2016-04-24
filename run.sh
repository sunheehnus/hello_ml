#!/bin/bash

gcc minor_LR.c
./a.out
python show_pic.py
rm a.out
