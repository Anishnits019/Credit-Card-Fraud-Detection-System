from pydantic import BaseModel,Field
from datetime import datetime
class TransactionModel(BaseModel):

    trans_date_trans_time: datetime
    cc_num: int
    merchant: str
    category: str
    amt: float

    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int

    lat: float
    long: float
    city_pop: int
    job: str
    dob: datetime
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: int
