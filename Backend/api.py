from fastapi import APIRouter,HTTPException
from typing import List

from .models import Transaction_Model
from .database import collection

router=APIRouter(
    prefix="/predict",
    tags=['predict']
)

@router.get(
        "/{cc_num}",
         response_model=List[Transaction_Model]
)

def get_cards(cc_num:int):
    cards= (
                  collection.
                  find({"cc_num": cc_num},{"_id":0}).\
                  sort("trans_date_trans_time")).limit(10)
    if not cards:
        raise HTTPException(
            status_Code=404,
            detail=f"No transactions found for card {cc_num}"
        )
    
    return list(cards)


