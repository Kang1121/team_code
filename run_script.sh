conda activate good

# Fix file locking disabled on this file system (/scratch) when reading hdf5
export HDF5_USE_FILE_LOCKING='FALSE'

python main.py -model xxxxx -fold 6 -gpu 0 -mixup &&
python main.py -model xxxxx -eps 3 -fold 6 -gpu 0 -mixup -outrm  &&
python main.py -model xxxxx -eps 5 -fold 6 -gpu 0 -mixup -outrm &&
python main.py -model xxxxx -eps 7 -fold 6 -gpu 0 -mixup -outrm &&
python main.py -model xxxxx -eps 9 -fold 6 -gpu 0 -mixup -outrm &&
python main.py -model xxxxx -eps 11 -fold 6 -gpu 0 -mixup -outrm &&
python main.py -model xxxxx -eps 13 -fold 6 -gpu 0 -mixup -outrm &&
python main.py -model xxxxx -eps 13 -fold 6 -gpu 0 -mixup -outrm

conda deactivate

wait
