#!bin/bash
mkdir dataset
cd dataset
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -O inria.tar
tar xf inria.tar
mv INRIAPerson inria
wget https://www.dropbox.com/s/fjt76tcl9flurpb/test_inria_label.zip?dl=1 -O test_label.zip
wget https://www.dropbox.com/s/q0riodfy6bk9esk/train_inria_label.zip?dl=1 -O train_label.zip
unzip -q test_label.zip
unzip -q train_label.zip
cp -r train_inria_label/* inria/Train/pos/
cp -r test_inria_label/* inria/Test/pos
