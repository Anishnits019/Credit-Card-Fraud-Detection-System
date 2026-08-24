from fastapi import FastAPI,Request,HTTPException,status
from fastapi.responses import JSONResponse
from Backend.models import get_data
CreditCardModel=get_data
app=FastAPI()
@app.get('/')
async def homepage():
 return ("message")
class CreditCardNotFoundError(Exception):
     def __init__(self,cc_num:int):
       self.cc_num=cc_num

@app.exception_handler(CreditCardNotFoundError)
async def credit_card_not_found_handler(request: Request, exc: CreditCardNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "status": "error",
            "error_code": "CARD_NOT_FOUND",
            "message": f"Credit card number {exc.cc_num} was not found in our database.",
        },
        headers={"X-Error-Source": "FraudEngine"}
    )
        
@app.get('/predict/{cc_num}/{amount}/{category}',resposne_model=CreditCardModel,status_code=status.HTTP_201_CREATED)
def predict_fraud(cc_num:int,amount:int,catogty:str):
 if cc_num not in items:
    raise HTTPException(
      status_code=404,
      detail='item not found'
    )
 custome_headers={
  'model-version':'lightgbm',
  'Cache-Control':"no-store"

 }
 return JSONResponse(content=response_data)


@app.post('/add/{cc_num}/{amount}/{category}')
def add_date(cc_num:int,amount:int,category:str):
 

 
