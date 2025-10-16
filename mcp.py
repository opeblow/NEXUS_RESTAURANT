from mcp.server.fastmcp import FastMCP
import random
import uuid
from datetime import datetime,timedelta
import time

mcp=FastMCP("Restaurant Payment and Delivery Service")

#Mock drivers
DRIVERS=[
    {"name":"obi","phone":"+234-000-900-0000","vehicle":"Motorcycle"},
    {"name":"segun","phone":"+123-480-000-0012","vehicle":"Car"},
    {"name":"Ali","phone":"+789-099-777-4444","vehicle":"Bike"},
    {"name":"Tunde","phone":"+111-500-430-3333","vecicle":"Car"}
]

#Mock stripe test cards
TEST_CARDS={
    "4242424242424242":{"status":"success","brand":"Visa"},
    "4000000000000002":{"status":"declined","reason":"card_declined"},
    "4000000000009995":{"status":"declined","reason":"insufficient_funds"}
}

#First mcp tool
@mcp.tool()
def process_payment(
    amount:float,
    customer_name:str,
    customer_email:str,
    card_number:"4242424242424242"
)->dict:
    """Process payment using stripe-style mock

    Args:
       amount:Total amount in USD
       customer_name:Customer's full name
       customer_email:Customer's email 
       card_number:Test card number(default:success)

    Returns:
        Payment result with transaction details
    
    """
    print(f"\n[MCP] Processing payment:${amount:.2f}for {customer_name}")
    time.sleep(1)

    card_info=TEST_CARDS.get(card_number,TEST_CARDS["4242424242424242"])
    if card_info["status"]=="success":
        charge_id=f"ch_{uuid.uuid().hex[:24]}"
        payment_intent_id=f"pi_{uuid.uuid4().hex[:24]}"

        result={
            "status":"succeeded",
            "id":charge_id,
            "payment_intent":payment_intent_id,
            "amount":int(amount*100),
            "amount_received":int(amount * 100),
            "currency":"Usd",
            "customer_name":customer_name,
            "customer_email":customer_email,
            "card_brand":card_info["brand"],
            "card_last4":card_number[-4:],
            "created":int(datetime.now().timestamp()),
            "receipt_url":f"https://stripe.com/receipts/{charge_id}",
            "message":f"Pyment successful! ${amount:.2f} charged to {card_info['brand']}"
        }

        print(f"[MCP] Payment succeeded:{charge_id}")
        return result
    else:
        result={
            "status":"failed",
            "error":{
                "type":"card_error",
                "code":card_info["reason"],
                "messgae":f"Your card was declined.{card_info['reason'].replace('_','').title()}"
            },
            "amount":int(amount *100),
            "message":f"Payment failed :{card_info['reason'].replace('_','').title()}"
        }

        print(f"[MCP] Payment failed:{card_info['reason']}")

#second mcp tool

@mcp.tool()
def assign_delivery_rider(
    order_id:str,
    delivery_address:str,
    customer_name:str,
    customer_phone:str,
    order_total:float
)->dict:
    """Assign delivery rider to order

    Args:
      order_id:Unique order identifier
      delivery_address:Full delivery address
      customer_name;Customer's name
      customer_phone:Customer's phone number
      order-total:Order total amount

    Returns:
      Rider assignment detils with tracking info

    """
    print(f"[MCP]Assigning rider for order:{order_id}")
    time.sleep(0.5)
    #select random rider
    rider=random.choice(DRIVERS)

    #calculate ETA(15-60 minutes)
    eta_minutes=random.randint(15,60)
    eta_time=datetime.now() + timedelta(minutes=eta_minutes)

    delivery_id=f"DEL{uuid.uuid4().hex[:8].upper()}"
    tracking_url=f"https://track.delivery.com/{delivery_id}"

    result={
        "status":"assigned",
        "delivery_id":delivery_id,
        "order_id":order_id,
        "rider":{
            "name":rider["name"],
            "phone":rider["phone"],
            "vehicle":rider["vehicle"]
        },
        "delivery_address":delivery_address,
        "customer_name":customer_name,
        "customer_phone":customer_phone,
        "order_total":order_total,
        "eta_minutes":eta_minutes,
        "estimated_arrival":eta_time,
        "tracking_url":tracking_url,
        "assigned_at":datetime.now().isoformat(),
        "message":f"Rider {rider['name']} assigned Vehicle:{rider['vehicle']}|ETA:{eta_minutes}|Track:{tracking_url}"

    }

    print(f"[MCP]Rider assigned:{rider['name']} (ETA:{eta_minutes}mins)")
    return result


#third mcp tool
@mcp.tool()
def get_delivery_status(delivery_id:str)->dict:
    """Get current delivery status
    
    Args:
      delivery_id:Delivery tracking ID

    Returns:
      Current delivery status and location
    
    """

    print(f"[MCP]Checking status for:{delivery_id}")

    statusses=[
        {"status":"confirmed","message":"Order confirmed","progress":10},
        {"status":"preparing","message":"Restaurant is preparing your order","progress":30},
        {"status":"ready","message":"Order is ready for pickup","progress":50},
        {"status":"picked_up","message":"Rider has picked up your order","progress":60},
        {"status":"in_transit","message":"Order is on the way","progress":80},
        {"status":"nearby","message":"Rider is 5 minutes away","progress":95},
        {"status":"delivered","message":"Order delivered successfully","progress":100}
    ]

    current_status=random.choice(statusses)

    result={
        "delivery_id":delivery_id,
        "status":current_status["status"],
        "message":f"{current_status['message']}",
        "progress_percentage":current_status["progress"],
        "last_updated":datetime.now().isoformat()
    }
    print(f"[MCP] Status:{current_status['status']}")
    return result


#fourth mcp tool

@mcp.tool()
def refund_payment(charge_id:str,amount:float,reason:str="customer_request")->dict:
    """Process refund for a payment
    Args:
      charge_id:Original charge ID
      amount:Amount to refund
      reason:Reason for refund
    
    Returns:
      Returns confirmation
    
    """

    print(f"[MCP]Processing refund:${amount:.2f}for {charge_id}")
    time.sleep(0.5)
    refund_id=f"re_{uuid.uuid4().hex[:24]}"

    result={
        "status":"succeeded",
        "id":refund_id,
        "charge":charge_id,
        "amount":int(amount*100),
        "currency":"Usd",
        "reason":reason,
        "created":int(datetime.now().timestamp()),
        "message":f"Refund of ${amount:.2f}processed successfully"
    }

    print(f"[MCP]Refund processed:{refund_id}")
    return result

if __name__=="__main__":
    print("Starting MCP Payment and Delivery Server.")
    print("Server ready at stdio")
    print("="*60)
    mcp.run()

