# Language: Python 3.9.7

**Virtual environment for my local system** :phd_project <br>
**VIRTUAL ENVIRONMENT SETUP**<br>
Install dependencies via Anaconda<br>
1.	Download and Install Anaconda (for ATU Company portal can be used) <br>
or<br>
https://www.anaconda.com/download<br>
2.	Open the command prompt and run the following command with environment.yml file (Shared the file)<br>
Use command:<br>
**conda env create --name <new_environment_name> --file environment.yml**<br>
Replace <new_environment_name> with the desired name for the new environment.<br>
This command will create a new environment using the specifications in the environment.yml file. Once, installed.<br>
3.	Activate the new environment<br>
**conda activate <new_environment_name>**<br>

4.	Verify the environment<br>
**conda list**<br>
This command will display the list of packages installed in the cloned environment, which should match the dependencies specified in your original environment.<br>
**conda env list**<br>
This command to check the environment.<br>

# EDA_datapreparation
The repository has hosted Python codes for Exploratory data analysis(EDA) and data preparation(Time series SST daily data) for the machine learning model, using the time series SST daily data from different sources. The list of sources is below:

**For SST data preparation four main sources are used**<br>
**1. Marine Institute:**<br>
https://erddap.marine.ie/erddap/info/index.html?page=1&itemsPerPage=1000 <br>
https://erddap3.marine.ie/erddap/info/index.html?page=1&itemsPerPage=1000 <br>
Using Underway vessel data collected every 10sec roughly from 1982-2022.<br>

**2. ICES data from different sources:**<br>
https://www.ices.dk/data/data-portals/Pages/ocean.aspx<br>

2.1	Bottle & Low resolution CTD data<br>
2.2	Expendable Bathythermograph data<br>
2.3	High-resolution CTD data<br>
2.4	Pump data<br>
2.5	Surface data<br>

ICES data are the data collected in different ways at the vertical profile of the ocean, but we will be using SST at the surface i.e. the depth==0.<br>
The MI and ICES are the base data, merging wherever there is a data gap, NOAA/Copernicus data will be used to fill the data gap as per the resolution used 0.25 or 0.05 

**3. NOAA SST data(0.25 &times; 0.25) ** <be>
**4. Copernicus SST data (0.05 &times; 0.05)**<br>
https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/ <br>
https://marine.copernicus.eu/<br>

**WORKFLOW for SST DATA PREPARATION**<br>
**Bounding box : [-14W, -5W, 49N, 56N] OR Region of Interest**<br>

**1.	Marine Institute Underway Vessel data**<br>
The original unedited csv data is **UW_data.csv**, this is not put in the repository as the file size is above the allowed size on Git Hub.<br>
However, files are put in the shared folder for the time being, https://drive.google.com/drive/folders/1HjIfXpZDB_681QWMvSE8O4A3fAB7iKRG <br>
**Code 1:** mi_filter_data1.py<br>
**Input Data:** UW_data.csv<br>
**Output data:** UW_data_r_Ireland.csv<br>
Code1 deals with:<br>
•	Elimination of non-needed attributes like Salinity<br>
•	Filtering data only with Bounding box<br>
•	Dropping the NaN values represented by -999.0<br>
•	Saving to a new file called UW_data_r_Ireland.csv<br>

As data is collected every 10sec, hence averaging the SST data for each day<br>
**Code2:** mi_eda2.py<br>
**Input data:** eez.shp (Exclusive economic zone shapefile), to keep the data points for the areas around Ireland that are under EEZ and<br>
UW_data_r_Ireland.csv<br>
**Bridge output:** daily_averages.csv<br>
This is the averaged file but no filtration using EEZ file, so it has estuaries and bays data which we want to eliminate therefore, we will consider EEZ<br>
**Output data:** daily_avgMI_eez.csv<br>
Code 2 deals with:<br>
•	Finding and elimination of duplicate values<br>
•	Grouping and averaging each day’s data as the data is collected every 10sec for each day<br>
•	Filtering data using EEZ shapefile<be>
**Code2.1:** mi_eda2_1.py<br>
**Input data:** daily_avgMI_eez.csv<br>
**Code2.2:** mi_outlier2_2.py<br>
**Input data:** daily_avgMI_eez.csv<br>
**Output data:** daily_avgMI_eez_f.csv<br>
Code2.2 removes the outlier and saves it as a new file.<br>


**2.	ICES SST data**<br>
Random point data taken over different latitude longitude positions across the water column,<br>
but as our base MI SST data, is only for surface i.e at depth 0<br>
Filtering of data with depth==0<br>
Also, ICES data from all sources are merged except the pump data as data was not available at depth==0<br>
Input data are: <br>
•	All_work_data/ICES/ICES_bottlelowresctd/bottle_lowres_ctd/bottle_lowres_temp.csv<br>
•	All_work_data/ICES/ICES_Expendabledatac/xbt_data/xbt_data_surfaceonly.csv<br>
•	All_work_data/ICES/ICES_high_resolution/high_res_ctd/high_res_data.csv<br>
•	All_work_data/ICES/ICES_OceanSurfacedat/Surface_data/oceansurfacedat_alter.csv<br>
These files are merged to form the ices_merged_data.csv, these are simply concatenated.<br>


**Code3:** ices_eda3.py<br>
**Input data:** ices_merged_data.csv<br>
Removal of outlier and using the shapefile EEZ only to consider EEZ area<br>
**Output data:** icesmerged_eez.csv<br>
Code3 deals with:<br>
•	Merging of different ICES files<br>
•	Removing the data with no values (here wherever there are no data values are blank) <br>
•	Removing the duplicate values (the first encounter of duplicate values is removed) <br>
•	Finding the outlier and plotting <br>


**3.	Merging of MI and ICES data**<br>
**Code4:** merge4.py<br>
**Input data:** icesmerged_eez.csv<br>
                daily_avgMI_eez_f.csv<br>
**Output data:** icesmi_eez_data.csv<br>
Code4 deals with:<br>
•	It merges the csv file from ICES and MI, using concatenation.<br>
•	Rounds off the longitude and latitude to two decimal places.<br>

Final Data Prepared from ICES and MI: icesmi_eez_data.csv (22103 counts)<br>


Now interpolation needs to be done one 0.25 and another 0.05 based on NOAA and Copernicus's different spatial resolution. <br>
**4. With reference to NOAA (0.25 resolution)** <br>
**Code5:** plot0_25.py<br>
It interpolates and plots the data of ICESMI merged data.<br>
**Code6:** interp_0.25work.py<br>
Code6 uses the IDW technique to fill the value in the grid.<br>
**Input file:** icesmi_eez_data.csv<br>
**Output file:** icesmigriddedsst_0.25data.nc<br>

Before code7<br>
**code6.1:** dttime_date.py<br>
It changes the DateTime to just the date on the NOAA data and saves it to final_noaasst_data.nc <br>
**Code6.2:** icesmi_alter.py<br>
It changes the difference in the way longitude numbers are in icesmi and noaa file<br>
**Input file:** icesmigriddedsst_0.25data.nc<br>
**Output file:** icesmigriddedsst_0.25data_standard.nc<br>
**Code7:** noaa_icesmi_combine.py<br>
It combines the data of icesmi NOAA by filling the nan values in icesmi with NOAA values.<br>
**Output file:** noaa_icesmi_combinefile.nc

**Used file for code 7:** <br>
input_file1 = 'Data_noaa_copernicus/noaa_avhrr/icesmigriddedsst_0.25data_standard.nc'<br>
input_file2 = 'Data_noaa_copernicus/noaa_avhrr/final_noaasst_data.nc' #This is the merged file of all the years of data from 1982 to 2022.<br>

**Final Prepared data using MI, ICES, and NOAA is** noaa_icesmi_combinefile.nc<br>

**5. With reference to Copernicus (0.05)** <br>

**Code 8:**copernicus_icesmi.py<br>
**Input files:** copernicus_icesmi_sst0.05data.nc<br>
                 Copernicus_sst_Celsius.nc<br>
**Output files:** coper_miicescombine.nc<br>
The above code combines the icesmi Copernicus at resolution 0.05 <br>
**Final Prepared data using MI, ICES, and Copernicus is** coper_miicescombine.nc <br>




