{\rtf1\ansi\ansicpg1252\cocoartf2707
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 conda activate good\
cd run \
\
# Fix file locking disabled on this file system (/scratch) when reading hdf5\
export HDF5_USE_FILE_LOCKING='FALSE'\
\
python main.py -model xxxxx -fold 6 -gpu 0 -mixup &&\
python main.py -model xxxxx -eps 3 -fold 6 -gpu 0 -mixup -outrm  &&\
python main.py -model xxxxx -eps 5 -fold 6 -gpu 0 -mixup -outrm &&\
python main.py -model xxxxx -eps 7 -fold 6 -gpu 0 -mixup -outrm &&\
python main.py -model xxxxx -eps 9 -fold 6 -gpu 0 -mixup -outrm &&\
python main.py -model xxxxx -eps 11 -fold 6 -gpu 0 -mixup -outrm &&\
python main.py -model xxxxx -eps 13 -fold 6 -gpu 0 -mixup -outrm &&\
python main.py -model xxxxx -eps 13 -fold 6 -gpu 0 -mixup -outrm \
\
\
cd ~\
conda deactivate\
\
wait}