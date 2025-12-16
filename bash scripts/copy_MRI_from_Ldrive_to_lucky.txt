# copy anatomical scans to Lucky3 for source reco using bash
find "~/LabData/Lab_LucaC/A_QNC_Databank/MRI_Baseline_Bids" -type f -path "*/anat/*.nii*" -exec cp {} /home/carinaf/tms_mdd/anat \;
