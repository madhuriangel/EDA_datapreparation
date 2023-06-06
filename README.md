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

**Virtual working environment Setup**<br>
Install dependencies via Anaconda<br>
1.	Download and Install Anaconda (for ATU Company portal can be used) <br>
2.	Open the command prompt and run the following command with environment.yml file (Shared the file)<br>
Use command:<br>
**conda env create --name <new_environment_name> --file environment.yml**<br>
Replace <new_environment_name> with the desired name for the new environment.<br>
This command will create a new environment using the specifications in the environment.yml file. Once, installed.
3.	Activate the new environment
conda activate <new_environment_name>

4.	Verify the environment
conda list
This command will display the list of packages installed in the cloned environment, which should match the dependencies specified in your original environment.
conda env list 
This command to check the environment.



