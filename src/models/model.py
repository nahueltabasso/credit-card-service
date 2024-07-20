from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class CreditCardData:
    payment_network: str = None
    card_number: str = None
    name: str = None
    expiration_date: datetime = None
    create_at: datetime = None
    obs: str = None
    
    def to_dict(self):
        data = asdict(self)
        if self.create_at:
            data['create_at'] = self.create_at.isoformat()
        return data