#!/usr/bin/env bash

sudo apt install -y -V apt-transport-https lsb-release
curl https://dist.apache.org/repos/dist/dev/arrow/KEYS | sudo apt-key add -
sudo tee /etc/apt/sources.list.d/apache-arrow.list <<APT_LINE
deb [arch=amd64] https://dl.bintray.com/apache/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/ $(lsb_release --codename --short) main
deb-src https://dl.bintray.com/apache/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/ $(lsb_release --codename --short) main
APT_LINE
sudo apt update --allow-insecure-repositories --allow-unauthenticated
sudo apt install -y -V --allow-unauthenticated apache-arrow
sudo apt install -y -V --allow-unauthenticated libarrow-dev
sudo apt install -y -V --allow-unauthenticated libarrow-dev_0.13
sudo apt install -y -V --allow-unauthenticated libparquet-dev
