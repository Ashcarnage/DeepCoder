#!/bin/bash
cd /DeepCoder
git add .
git commit -m "Auto-update from SSH $(date)"
git push origin main