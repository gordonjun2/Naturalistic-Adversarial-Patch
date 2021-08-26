#!/bin/bash
wget https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar
# mv pretrained_biggan.tar ./GANLatentDiscovery/models/
tar xf pretrained_biggan.tar 
mv ./pretrained ./GANLatentDiscovery/models/
rm pretrained_biggan.tar
