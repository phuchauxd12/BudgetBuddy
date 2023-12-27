def bankEntity(item) -> dict:
    return  {
        'id': str(item["_id"]),
        'date': item['date'],
        'total_income': item['total_income'],
        'total_expenses': item['total_expenses']
    }
    
def banksEntity(entity)-> list:
    return[bankEntity(item) for item in entity ]

