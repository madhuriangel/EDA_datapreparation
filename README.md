# EDA_datapreparation
Exploratory data analysis and preparation of data for ML model

**For SST data preparation three main sources to be used**
1. MI underway SST data
2. ICES data
3. NOAA SST data(0.25 &times; 0.25) and Copernicus SST data (0.05 &times; 0.05)

**Step 1**<br>
Analysis and explore the Marine institute SST data
MI underway SST data 
Temporal resolution:10sec, surface of the ocean.
For Marine heatwave daily resolution is needed, averaging each day 10sec data to daily data.

**Step 2**<br>
Use the ICES data to do the gap filling 
As underway data is of the surface of ocean(depth=0), therefore only data taken at depth=0 is considered
1. Bottle low resolution data
2. XBT data
3. High resolution data
4. Pump data (Temperature data is all blank at depth=0), hence eliminated
5. Ocean Surface data<br>
Jupyter code file name: ices_datamerge<br>
All the above sources are merged(excluding pump data)<br>


**Step 2.1**<br>
After this MI and ICES data are merged(merged_data\\icesmi_merged_data.csv)<br>
Jupyter code file name: icesmi_merge


**Step 3**<br>
Use NOAA or Copernicus data to do the gap filling based on the grid we want 0.25 &times; 0.25 or 0.05 &times; 0.05


