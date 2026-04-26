# sequence_landscape_align
Script for aligning construction montage undergoing drastic change for images taken in the same location. 
Input images are expected to labeled as 00001, 00002, ..., 0000n. 
FLANN was used to avoid importing another package, but FAISS is preferred. 
Detector/Descriptor being used is RootSIFT.
