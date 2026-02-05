from fastapi import FastAPI, Request, HTTPException

import os
import glob

from schemas import PreprocessStatusResponse, ErrorDetail, JobCreatedResponse, PreprocessRequest, OutputDir, DatasetDescription, PreprocessingConfig, JobStatus

app = FastAPI()



"""""
Current API endpoints:
1. endpoint to preprocess data
2. endpoint to upload new data to help calibrate the model
3. /stream so we can use the model that we trained
4. /epochs to basically split the data into epochs
5. /train to train the data
6. /change-feature to basically play around with the feature



"""

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}



#pre process the dataset, and return a job id so we can keep track of it
@app.post("/preprocess")
def preprocess_data(request: PreprocessRequest):
    raw_dir = request.DatasetDescription.raw_dir
    out_dir = request.OutputDir.out_dir

    if os.path.exists(raw_dir):
        if not (glob.glob(os.path.join(raw_dir, "*PSG.edf")) and glob.glob(os.path.join(raw_dir, "*Hypnogram.edf"))):
            raise HTTPException(status_code= 404, detail = "No PSG and hypongram files found")
        if not glob.glob(os.path.join(raw_dir, "*PSG.edf")):
            raise HTTPException(status_code=404, detail = "No PSG files found")
        if not glob.glob(os.path.exists(raw_dir, "*Hypnogram.edf")):
            raise HTTPException(status_code=404, detail = "No Hypnogram files found")



       
    else:
        raise HTTPException(status_code=404, detail="Not a valid file path for raw directory")
    
    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404, detail="Output directory not a valid path")
    


    





#get info about that specific job id

@app.get("/get-preprocess/{job_id}")
def get_preprocess_status(job_id: str, request: Request):


                          



