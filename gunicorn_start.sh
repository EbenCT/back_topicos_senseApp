#!/bin/bash
echo "Iniciando servidor Gunicorn..."
exec gunicorn -w 4 -b 0.0.0.0:5000 image_server:app
