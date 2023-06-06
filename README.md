# EDA_datapreparation
The repository has hosts Python codes for Exploratory data analysis(EDA) and data (Time series SST daily data) preparation for machine learning model, using the time series SST daily data from different sources, list of sources are below:

**For SST data preparation three main sources to be used**<br>
**1.	Marine Institute:**<br>
https://erddap.marine.ie/erddap/info/index.html?page=1&itemsPerPage=1000 <br>
https://erddap3.marine.ie/erddap/info/index.html?page=1&itemsPerPage=1000 <br>
Using Underway vessel data collected every 10sec roughly from 1982-2022.<br>

**2.	ICES data from different sources:**<br>
https://www.ices.dk/data/data-portals/Pages/ocean.aspx<br>

2.1	Bottle & Low resolution CTD data<br>
2.2	Expendable Bathythermograph data<br>
2.3	High resolution CTD data<br>
2.4	Pump data<br>
2.5	Surface data<br>

ICES data are the data collected by different ways at the vertical profile of the ocean, but we will be using SST at the surface i.e the depth==0.<br>
The MI and ICES are the base data, after merging where ever there is a data gap, NOAA/Copernicus data will be used to fill the data gap as per the resolution used 0.25 or 0.05 

**3. NOAA SST data(0.25 &times; 0.25) and Copernicus SST data (0.05 &times; 0.05)**
https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/ <br>
https://marine.copernicus.eu/<br>


**VIRTUAL ENVIRONMENT SETUP**<br>
Install dependencies via Anaconda<br>
1.	Download and Install Anaconda (for ATU Company portal can be used) <br>
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

**WORKFLOW for SST DATA PREPARATION**<br>
**Bounding box : [-14W, -5W, 49N, 56N]**<br>
**1.	Marine Institute Underway Vessel data**<br>
First filtering of data for the chosen bounding box<br>
Dropping the Salinity Column and Temperature== -999.0<br>
**Code:** mi_filter_data1.py<br>
**Input Data:** UW_data.csv<br>
**Output data:** UW_data_r_Ireland.csv<br>

As data is collected every 10sec, hence averaging the SST data for each day<br>
Elimination of null values<br>
Elimination of duplicate values<br>
Elimination of outlier<br>
**Code:** mi_eda2.py<br>
**Input data:** eez.shp (Exclusive economic zone shapefile), to just keep the data points for the areas around Ireland which is under EEZ and
UW_data_r_Ireland.csv<br>
**Output data:** daily_avgMI_eez.csv<br>

**2.	ICES SST data**<br>
Random point data taken over different latitude longitude position across the water column,<br>
but as our base MI SST data, is only for surface i.e at depth 0<br>
Filtering of data with depth==0<br>
And also ICES data from all sources are merged except the pump data as data was not available at depth==0<br>
Elimination of null values<br>
Elimination of duplicate values<br>
**Code:** ices_eda3.py<br>
**Input data:** ices_merged_cleaned.csv<br>
Removal of outlier and using the shapefile EEZ to only consider EEZ area<br>
**Output data:** icesmerged_eez.csv<br>

**3.	Merging of MI and ICES data**<br>
**Code:** merge4.py<br>
**Input data:** icesmerged_eez.csv<br>
                daily_avgMI_eez.csv<br>
**Output data:** icesmi_eez_data.csv<br>
Merging and rounding off longitude latitude to two decimal points.<br>

There are few other codes<br>
icesmi_dotplot.py (this is a scatter plot for the merged data)<br>
plot0_25.py (Plot with resolution of 0.25)<br>
plot0_05.py (Plot with resolution of 0.05)<br>
plot0_25_each.py (Plot for each mm/year with resolution 0.25)<br>
Will create a function later on.<br>

**4.	Gridding of data as per the resolution, and using interpolation to get the values over the grid, where data points are present.**<br>
**Code:** interp_trial.py<br>
**Input data:** icesmi_eez_data.csv<br>

**5.	NOAA (0.25)/Copernicus (0.05)**<br>

As per the selected resolution fill the data using NOAA/Copernicus data




