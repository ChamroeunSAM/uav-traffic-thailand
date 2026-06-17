# Code and Data Availability

To support reproducibility, all source code for the detector benchmark,
tracker evaluation, calibration, and trajectory-quality framework, together
with the aggregated result files and the scripts that regenerate every table
and figure in this paper, is publicly available at
https://github.com/ChamroeunSAM/uav-traffic-thailand. The repository
documents the full five-stage methodology, the locked software environment
(`requirements.lock.txt`, PyTorch 2.11.0 + CUDA 12.8), and the fixed random
seeds used throughout, so that the reported experiments can be re-run
end-to-end on equivalent hardware. The complete record of architectural
decisions is provided in the repository's decision log.
