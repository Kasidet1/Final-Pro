import uvicorn
from fastapi import FastAPI, HTTPException, Query
from joblib import load
import pandas as pd
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI(title="Job Satisfaction Prediction API",
              description="API for Predicting Employee Job Satisfaction",
              version="1.0")

# Load models and preprocessor
model_path = 'D:/COE64-335/Final Pro/FastAPI/best_random_forest_model.pkl'
preprocessor_path = 'D:/COE64-335/Final Pro/FastAPI/preprocessor.pkl'

try:
    model = load(model_path)
    preprocessor = load(preprocessor_path)
    logging.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or preprocessor: {e}")
    raise HTTPException(status_code=500, detail="Error loading model or preprocessor")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Job Satisfaction Prediction API!"}


@app.post('/predict_job_satisfaction', tags=["predictions"])
async def predict_job_satisfaction(
        Gender: str = Query(..., description="เพศของพนักงาน (e.g., 'Male', 'Female')"),
        MaritalStatus: str = Query(..., description="สถานภาพสมรสของพนักงาน (e.g., 'Single', 'Married')"),
        Dept: str = Query(..., description="แผนกที่พนักงานทำงานอยู่ (e.g., 'HR', 'Sales')"),
        EmpType: str = Query(..., description="ประเภทการจ้างงานของพนักงาน (e.g., 'Full-Time', 'Part-Time')"),
        CommuteMode: str = Query(..., description="วิธีการเดินทางของพนักงาน (e.g., 'Car', 'Public Transit')"),
        EduLevel: str = Query(..., description="ระดับการศึกษาของพนักงาน (e.g., 'Bachelor', 'Master')"),
        JobLevel: str = Query(...,
                              description="ระดับงานของพนักงาน (e.g., 'Mid', 'Intern/Fresher', 'Junior', 'Senior', 'Lead')"),
        Stress: int = Query(..., description="ระดับความเครียด (e.g., 0, 1, 2)"),
        WorkEnv: int = Query(..., description="สภาพแวดล้อมการทำงาน (1-5)"),
        Age: int = Query(..., description="อายุของพนักงาน (e.g., 25, 30)"),
        TeamSize: int = Query(..., description="ขนาดของทีมที่พนักงานทำงานอยู่ (e.g., 3, 10)"),
        haveOT: str = Query(..., description="การทำงานล่วงเวลา (e.g., 'Yes', 'No')"),
        Workload: int = Query(..., description="ภาระงาน (1-5)"),
        TrainingHoursPerYear: float = Query(..., description="ชั่วโมงการฝึกอบรมต่อปี (e.g., 10.5, 20.0)"),
        WLB: int = Query(..., description="สมดุลระหว่างงานและชีวิต (1-5)"),
        SleepHours: float = Query(..., description="ชั่วโมงการนอน (e.g., 6.5, 8.0)"),
        Experience: int = Query(..., description="จำนวนปีของประสบการณ์ในการทำงาน (e.g., 2, 5)"),
        NumReports: int = Query(..., description="จำนวนรายงานที่พนักงานรับผิดชอบ (e.g., 1, 5)"),
        CommuteDistance: int = Query(..., description="ระยะทางการเดินทาง (e.g., 5, 15)"),
        NumCompanies: int = Query(..., description="จำนวนบริษัทที่พนักงานเคยทำงาน (e.g., 1, 3)"),
        PhysicalActivityHours: float = Query(..., description="ชั่วโมงกิจกรรมทางกายต่อสัปดาห์ (e.g., 1.0, 4.5)")
):
    try:
        # Validate JobLevel
        valid_job_levels = {'Mid', 'Intern/Fresher', 'Junior', 'Senior', 'Lead'}
        if JobLevel not in valid_job_levels:
            raise HTTPException(status_code=400,
                                detail="Invalid JobLevel. Must be one of: Mid, Intern/Fresher, Junior, Senior, Lead.")

        # Validate WorkEnv
        if not (1 <= WorkEnv <= 5):
            raise HTTPException(status_code=400, detail="WorkEnv must be an integer between 1 and 5.")

        # Validate Workload
        if not (1 <= Workload <= 5):
            raise HTTPException(status_code=400, detail="Workload must be an integer between 1 and 5.")

        # Validate WLB
        if not (1 <= WLB <= 5):
            raise HTTPException(status_code=400, detail="WLB must be an integer between 1 and 5.")

        # Validate TrainingHoursPerYear, SleepHours, PhysicalActivityHours
        if TrainingHoursPerYear < 0:
            raise HTTPException(status_code=400, detail="TrainingHoursPerYear must be a non-negative number.")
        if SleepHours < 0:
            raise HTTPException(status_code=400, detail="SleepHours must be a non-negative number.")
        if PhysicalActivityHours < 0:
            raise HTTPException(status_code=400, detail="PhysicalActivityHours must be a non-negative number.")

        # เตรียมข้อมูลนำเข้าพร้อมคอลัมน์ที่ต้องการทั้งหมด
        input_data = pd.DataFrame({
            'Gender': [Gender],
            'MaritalStatus': [MaritalStatus],
            'Dept': [Dept],
            'EmpType': [EmpType],
            'CommuteMode': [CommuteMode],
            'EduLevel': [EduLevel],
            'JobLevel': [JobLevel],
            'Stress': [Stress],
            'WorkEnv': [WorkEnv],
            'Age': [Age],
            'TeamSize': [TeamSize],
            'haveOT': [haveOT],
            'Workload': [Workload],
            'TrainingHoursPerYear': [TrainingHoursPerYear],
            'WLB': [WLB],
            'SleepHours': [SleepHours],
            'Experience': [Experience],
            'NumReports': [NumReports],
            'CommuteDistance': [CommuteDistance],
            'NumCompanies': [NumCompanies],
            'PhysicalActivityHours': [PhysicalActivityHours]
        })

        # แสดงประเภทข้อมูลของ DataFrame ก่อนการแปลง
        logging.info(f"ประเภทข้อมูลก่อนการแปลง: {input_data.dtypes}")

        # แปลงคอลัมน์ที่เป็นตัวเลข
        numeric_columns = ['JobLevel', 'Age', 'TeamSize', 'TrainingHoursPerYear', 'SleepHours',
                           'Experience', 'NumReports', 'CommuteDistance', 'NumCompanies', 'PhysicalActivityHours']

        for col in numeric_columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        # แสดงประเภทข้อมูลของ DataFrame หลังการแปลง
        logging.info(f"ประเภทข้อมูลหลังการแปลง: {input_data.dtypes}")

        # จัดการกับค่า NaN (ถ้ามีหลังจากการแปลง)
        if input_data.isnull().values.any():
            logging.warning("พบค่า NaN ในข้อมูลนำเข้าหลังการแปลง. เติมค่าด้วยค่าที่ตั้งค่าเริ่มต้น.")
            input_data = input_data.fillna(0)

        # แปลงคอลัมน์ที่เป็นข้อมูลประเภทหมวดหมู่เป็นสตริงเพื่อป้องกันปัญหาประเภทข้อมูล
        categorical_columns = ['Gender', 'MaritalStatus', 'Dept', 'EmpType', 'CommuteMode', 'EduLevel', 'WorkEnv',
                               'haveOT', 'WLB']
        for col in categorical_columns:
            input_data[col] = input_data[col].astype(str)

        # แสดงข้อมูลนำเข้าสำหรับพรีโปรเซสเซอร์
        logging.info(f"ข้อมูลนำเข้าสำหรับพรีโปรเซสเซอร์: {input_data}")

        # ใช้พรีโปรเซสเซอร์เพื่อแปลงข้อมูลนำเข้า
        try:
            processed_data = preprocessor.transform(input_data)
        except Exception as e:
            logging.error(f"ข้อผิดพลาดในการแปลงข้อมูลโดยใช้พรีโปรเซสเซอร์: {e}")
            raise HTTPException(status_code=500, detail="ข้อผิดพลาดในการแปลงข้อมูล")

        # แสดงข้อมูลหลังการแปลง
        logging.info(f"ข้อมูลหลังการแปลง: {processed_data}")

        # ทำการทำนายโดยใช้โมเดล
        try:
            prediction = model.predict(processed_data).tolist()
        except Exception as e:
            logging.error(f"ข้อผิดพลาดในการทำนาย: {e}")
            raise HTTPException(status_code=500, detail="ข้อผิดพลาดในการทำนาย")

        return {"predicted_job_satisfaction": prediction[0]}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"ข้อผิดพลาดทั่วไป: {e}")
        raise HTTPException(status_code=500, detail="ข้อผิดพลาดภายในเซิร์ฟเวอร์")


if __name__ == "__main__":
    uvicorn.run("main:app", host="10.10.10.240", port=5000, reload=True)
