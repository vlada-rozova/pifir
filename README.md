# Detection of invasive fungal infection in PET/CT reports

### Data
We make use of the [PET/CT Invasive Fungal Infection Reports (PIFIR)](https://www.physionet.org/) dataset available at PhysioNet. Please note, the dataset is available under credentialed access and requires the user to sign the PhysioNet Credentialed Health Data Use Agreement. 

File overview:
* The folder "reports" contains the original de-identified reports in the .txt file format.
* The folder "annotations" contains brat annotation files in the .ann file format.
* The file "annotation.conf" is a configuration file for brat that defines concept categories and relations.
* The file "pilfer_metadata.csv" contains identifiers and a suggestion on how the data should be split for testing and validation.

The jupyter notebook `load-pifir.ipynb` allows the user to load the dataset and link reports with annotations.
