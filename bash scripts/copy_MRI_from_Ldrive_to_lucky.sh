# copy anatomical scans to neuroserv for source reconstruction
find "~/LabData/Lab_LucaC/A_QNC_Databank/MRI_Baseline_Bids" -type f -path "*/anat/*.nii*" -exec cp {} /home/carinaf/tms_mdd/anat \;
