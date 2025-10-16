import os
import json
import uuid
import datetime
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from dateutil.parser import parse
from dateutil import tz
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_tools_agent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import re
import traceback
import subprocess
import json as json_lib

# Load environment variables from .env file
load_dotenv()

# configuration
DB_FILE = "bookings.json"
CSV_PATH = "restaurantmenuchanges.csv"
FAISS_INDEX_PATH = "faiss_restaurant_index"
TABLE_CAPACITIES = {
    "T1": 2, "T2": 4, "T3": 6, "T4": 4, "T5": 8
}

# Standard FAQs and Policies
FAQ_AND_POLICIES_TEXT = [
    "Opening Hours: We are open from 11:00 AM to 10:00 PM, Tuesday to Sunday. We are closed on Mondays.",
    "Reservation Policy: Reservations can be made up to 30 days in advance. For parties larger than 8, please call us directly.",
    "Dietary Restrictions: We can accommodate most gluten-free and nut-free requests. All vegan items are clearly marked.",
    "Parking: Street parking is available, and there is a paid parking garage 2 blocks away.",
    "Wait Times: Standard wait time for a walk-in during peak hours (6 PM - 8 PM) is 30-45 minutes.",
    "Wine List: We offer a selection of local and imported red and white wines, detailed on our wine list.",
]

# Global variables to cache loaded data
_vector_store_cache = None
_menu_texts_cache = None
mcp_available=False
#  Loading and Formating the CSV Dataset 
def create_data_strings_from_csv(csv_path=CSV_PATH) -> List[str]:
    global _menu_texts_cache
    
    # Return cached data if already loaded
    if _menu_texts_cache is not None:
        return _menu_texts_cache
    
    print(f" Loading and processing menu data from {csv_path} ")
    try:
        df = pd.read_csv(csv_path,on_bad_lines="skip",engine="python")
        required_columns = ['menuItemName', 'menuItemDescription', 'menuItemCategory', 'menuItemCurrentPrice']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        menu_texts = df.apply(
            lambda row: f"{row['menuItemName']} - {row['menuItemDescription']} (Category: {row['menuItemCategory']}, Price: {row['menuItemCurrentPrice']})",
            axis=1
        ).tolist()
        print(f"Successfully loaded {len(menu_texts)} menu items from CSV")
        _menu_texts_cache = menu_texts
        return menu_texts
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at path: {csv_path}")
        print("Returning placeholder data. Please create 'menu.csv' and run again.")
        placeholder = ["Placeholder Dish - This is a placeholder description (Category: Placeholder, Price: $0.00)"]
        _menu_texts_cache = placeholder
        return placeholder
    except KeyError as e:
        print(f"ERROR: {e}")
        print("Ensure your CSV has 'menuItemName', 'menuItemDescription', 'menuItemCategory', and 'menuItemCurrentPrices' columns.")
        placeholder = ["Placeholder Dish - Check CSV column names (Category: Error, Price: $0.00)"]
        _menu_texts_cache = placeholder
        return placeholder

#  Combine and Chunk Documents 
def create_or_load_faiss_index(menu_texts: List[str]):
    global _vector_store_cache
    
    # Return cached vector store if already loaded
    if _vector_store_cache is not None:
        return _vector_store_cache
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading HuggingFace model: {e}")
        raise
    
    full_data = FAQ_AND_POLICIES_TEXT + menu_texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    combined_text = "\n".join(full_data)
    chunked_texts = text_splitter.split_text(combined_text)
    chunked_texts = [chunk.strip() for chunk in chunked_texts if chunk.strip()]
    print(f"Created {len(chunked_texts)} chunks from {len(full_data)} original documents.")

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index")
        vector_store = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f" FAISS index loaded successfully from {FAISS_INDEX_PATH}")
    else:
        print("Creating new FAISS index from chunked texts and saving locally")
        vector_store = FAISS.from_texts(chunked_texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index created and saved successfully to {FAISS_INDEX_PATH}")
    
    _vector_store_cache = vector_store
    return vector_store

#Definition of FAISS Semantic Search Tool
@tool
def faiss_semantic_search(query: str) -> str:
    """Performs a semantic search on the restaurant's menu and policies using FAISS."""
    vector_store = create_or_load_faiss_index(create_data_strings_from_csv())
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([result.page_content for result in results])

#  Initializing Local JSON Database 
def initialize_bookings_db():
    initial_db = {
        "tables": {table_id: {"capacity": capacity, "status": "available"} for table_id, capacity in TABLE_CAPACITIES.items()},
        "bookings": [],
        "orders": []
    }
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump(initial_db, f, indent=2)
        print(f"Initialized {DB_FILE} with default structure.")
    else:
        print(f" Database {DB_FILE} already exists.")

#  Defining Utility Tools 
@tool
def mock_pos_sync(table_id: str, booking_details: dict) -> str:
    """Simulate updating the Point-Of-Sale(pos)system with a table booking."""
    return f"POS system updated for Table {table_id} with details: {booking_details}"

@tool
def send_reminder(booking_details: dict) -> str:
    """Simulate sending a reminder for a booking."""
    return f"Reminder sent for booking: {booking_details}"

@tool
def get_menu_items(query: str = "") -> str:
    """Retrieves menu items based on a query or returns all menu items if no query is provided."""
    vector_store = create_or_load_faiss_index(create_data_strings_from_csv())
    if query:
        results = vector_store.similarity_search(query, k=5)
        return "\n".join([result.page_content for result in results])
    else:
        menu_texts = create_data_strings_from_csv()
        return "\n".join(menu_texts[:10])

@tool
def create_delivery_order(items: List[str], delivery_address: str, delivery_time: str) -> str:
    """Creates a delivery order with the specified items, address, and time."""
    try:
        with open(DB_FILE, "r") as f:
            db = json.load(f)
    except FileNotFoundError:
        initialize_bookings_db()
        with open(DB_FILE, "r") as f:
            db = json.load(f)
    
    menu_texts = create_data_strings_from_csv()
    valid_items = []
    for item in items:
        if any(item.lower() in menu_text.lower() for menu_text in menu_texts):
            valid_items.append(item)
        else:
            return f"Error: '{item}' is not on the menu. Please check the menu and try again."
    
    if not valid_items:
        return "Error: No valid menu items provided."
    
    try:
        wat_tz = tz.gettz("Africa/Lagos")
        parsed_time = parse(delivery_time, fuzzy=True, default=datetime.datetime.now(tz=wat_tz))
        formatted_time = parsed_time.isoformat()
    except Exception:
        return "Error: Invalid delivery time format. Please specify a time (e.g., '7 PM today')."
    
    order_id = str(uuid.uuid4())
    order_details = {
        "order_id": order_id,
        "items": valid_items,
        "delivery_address": delivery_address,
        "delivery_time": formatted_time,
        "status": "pending"
    }
    db["orders"].append(order_details)
    
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)
    
    delivery_result = mock_delivery_system.invoke({"order_details":order_details})
    return f"Order confirmed! Order ID: {order_id}\nItems: {', '.join(valid_items)}\nDelivery to: {delivery_address} at {formatted_time}\n{delivery_result}"

@tool
def mock_delivery_system(order_details: dict) -> str:
    """Simulates sending order details to a delivery service."""
    return f"Delivery system notified for Order ID: {order_details['order_id']} to {order_details['delivery_address']}"

# Creating Table Availability Tool 
@tool
def check_table_availability(party_size: int, time: str) -> str:
    """Find the most suitable available table for a given party size and time."""
    try:
        with open(DB_FILE, "r") as f:
            db = json.load(f)
    except FileNotFoundError:
        initialize_bookings_db()
        with open(DB_FILE, "r") as f:
            db = json.load(f)
    
    best_table = None
    min_capacity = float("inf")
    for table_id, table_info in db["tables"].items():
        if table_info["status"] == "available" and table_info["capacity"] >= party_size:
            if table_info["capacity"] < min_capacity:
                min_capacity = table_info["capacity"]
                best_table = table_id
    if best_table:
        return best_table
    return "No available tables for the requested party size and time."

#  Mock Google Calendar Tool 
@tool
def book_google_calendar_event(party_size:int,time:str ) -> str:
    """Simulate creating a Google Calendar event for a table booking.
    Args:
      party_size=Number of people in the party
      time:
         ISO formatted datetime string for the reservation.
    """
    event_id=str(uuid.uuid4())
    return f"Google Calendar event created with EVENT ID:{event_id} for {party_size} people at {time}"
  

# Creating Local Reservation Tool 
@tool
def reserve_table_locally(table_id: str, event_id: str, party_size:str,time:str) -> str:
    """Reserve a table in the local JSON database and trigger downstream mocks
    Args:
      table_id:The ID of the table to reserve
      evebt_id:The Google calendar event ID
      party_size:Number of people in the party
      time:ISO formatted datetime
    """
    try:
        with open(DB_FILE,"r")as file:
            db=json.load(file)
    except FileNotFoundError:
        initialize_bookings_db()
        with open(DB_FILE,"r")as file:
            db=json.load(file)
    booking_details={
        "party_size":party_size,
        "time":time,
        "event_id":event_id,
        "table_id":table_id
    }
    if table_id in db["tables"]:
        db["tables"][table_id]["status"]="occupied"
    db["bookings"].append(booking_details)

    with open(DB_FILE,"w")as file:
        json.dump(db,file,indent=2)

    pos_result=mock_pos_sync.invoke({"table_id":table_id,"booking_details":booking_details})
    reminder_result=send_reminder.invoke({"booking_details":booking_details})
    return f" Reservation Confirmed for Table {table_id}.Event ID:{event_id}\n{pos_result}\n{reminder_result}"


@tool
def process_payment_via_mcp(amount:float,customer_name:str,customer_email:str)->str:
    """Process payment through MCP server(stripe-style mock)"""
    if not mcp_available:
        return "Pyment system unavailable.Please try again later."
    try:
        mcp_input={
            "amount":amount,
            "customer_name":customer_name,
            "customer_email":customer_email,
            "card_number":"4242424242424242"
        }

        result=subprocess.run(
            ['python','c',
            f'from mcp import mcp;impor json;result=mcp.call_tool("process_payment",{json_lib.dumps(mcp_input)});print(json.dumps(result))' ],
            text=True,
            timeout=10
        )

        if result.returncode==0:
            response=json_lib.loads(result.stdout)
            return response.get("message","payment processed")
        else:
            return f"Payment failed:{result.stderr}"
    except Exception as e:
        return f" Payment failed :{str(e)}"
    
@tool
def assign_delivery_via_mcp(order_id:str,address:str,customer_name:str,phone:str,total:float)->str:
    """Assign delivery rider through MCP server."""
    if not mcp_available:
        return "Delivery system unavailable.Please try again later."
    try:
        mcp_input={
            "order_id":order_id,
            "delivery_address":address,
            "customer_name":customer_name,
            "customer_phone":phone,
            "order_total":total
        }

        result=subprocess.run(
            [ 'python','c',
              f'from mcp import mcp;import json;result=mcp.call_tool("assign_delivery_rider",{json_lib.dumps(mcp_input)});print(json.dumps(result))'

            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode==0:
            response=json_lib.loads(result.stdout)
            return response.get("message","Rider assigned")
        else:
            return f"Rider assignment failed:{result.stderr}"
        
    except Exception as e:
        return f" Rider assignment failed:{str(e)}"

#  Initializing LLM and Toolset 
def initialize_llm_and_tools():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please set it.")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    tools = [
        faiss_semantic_search,
        get_menu_items,
        check_table_availability,
        book_google_calendar_event,
        reserve_table_locally,
        create_delivery_order,
        mock_pos_sync,
        send_reminder,
        mock_delivery_system,
        process_payment_via_mcp,
        assign_delivery_via_mcp
    ]
    return llm, tools
#  Defining Agent Strategy 
def create_agent_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are NEXUS , a restaurant reservation and menu assistant, now also handling food delivery orders. For menu or policy queries, use get_menu_items or faiss_semantic_search.For each menu item,explicitly list the 'menuItemName',its 'menuItemCategory',and its 'menuItemCurrentPrice'.Do not omit or summarize these details.
          For booking requests, follow this sequence:
1. Call check_table_availability with party_size to find an available table.
2. If a table is available, call book_google_calendar_event with party_size and time to create a calendar event.Extract the event_id from the response.
3. Call reserve_table_locally with table_id,event_id,party_size,and time to finalize the booking,which will trigger mock_pos_sync and send_reminder.
For food delivery orders(2 workflows):
workflow(A)-A simple Delivey(No payments):
1.use get_menu+items to verify items.
2.call create_delivery_order with items,address,time(triggers mock_delivery_system)

workflow B-Delivery with payment
 1.when user asks about food(e.g,"jollof rice"),use get_menu_items
  2.show all matching items with prices
  3.Ask:would you like to order[item]for $? I can process payment and arrange delivery.
  4.If yes to payment +delivery:
         a.collect:name,email,address,phone
         b.call process_payment_via_mcp(amount,name,email)
         c.if mpayment successful,call assign_delivery_via_mcp(order_id,address,name,phone,total)
         d.Share rider details and ETA
         Use workflow A for simple orders.Use workflow B when user wants to pay
    5.If payments fails,suggest trying again.       """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return prompt

#  Initializing data during startup 
def initialize_data():
    """Pre-load CSV and FAISS index to show initialization messages."""
    print("\n" + "="*60)
    print("INITIALIZING RESTAURANT ASSISTANT")
    print("="*60)
    
    # Initializing database
    initialize_bookings_db()
    
    # Loading CSV data
    menu_texts = create_data_strings_from_csv()
    
    # Load or create FAISS index
    create_or_load_faiss_index(menu_texts)
    
    print("="*60)
    print("INITIALIZATION COMPLETE")
    print("="*60 + "\n")


def extract_party_size(text:str)->int:
    """Extract party size from text using regex-handles all format"""
    all_numbers=re.findall(r'\d+',text)
    patterns=[
        (r'(?:table|party|reservation)\s+(?:for|of)\s+(\d+)','table/party context'),
        (r'(?:for|of)\s+(\d+)\s*(?:people|persons|guests|ppl)?','for/of pattern'),
        (r'(\d+)\s+(?:people|persons|guests|ppl)','number + people'),

    ]
    text_lower=text.lower()

    for pattern,description in patterns:
        match=re.search(pattern,text_lower)
        if match:
            return int(match.group(1))
        
    if all_numbers:
        for num_str in all_numbers:
            num=int(num_str)
            if 1 <=num <=20 and num not in [11,12]:
                return num
        return int(all_numbers[0])
    return None
    

#  Enhanced Terminal Interface with Timezone Handling 
def main():
    # Initialize all data first
    initialize_data()
    
    try:
        llm, tools = initialize_llm_and_tools()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    prompt = create_agent_prompt()
    agent=(
        {
            "input":lambda x:(
                x["input"]
                if isinstance(x,dict)
                else(x[0]["input"] if isinstance(x,list)and len(x)>0 and isinstance(x[0],dict)and "input" in x[0]else None)
            ),
            "chat_history":lambda x:(
                x.get("chat_history",[])
                if isinstance(x,dict)
                else[]
            ),
            "agent_scratchpad":lambda x: (
                format_to_openai_tool_messages(
                    x.get("intermediate_steps",[])
                    if isinstance(x,dict)
                    else []
                )
                if isinstance(x,(dict,list))
                else{}
            ),   
        }
        |prompt
        |llm.bind_tools(tools)
        |OpenAIToolsAgentOutputParser()
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    agent_executor=AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    
    print("\n Welcome to NEXUS Restaurant Assistant! ")
    print("I can help with menu questions, restaurant policies, booking a table, or ordering food for delivery.")
    print("Examples:")
    print("  - Menu: 'What vegan options do you have?'")
    print("  - Booking: 'Book a table for 4 at 7 PM tomorrow'")
    print("  - Ordering: 'Order a Margherita Pizza and Caesar Salad for delivery to 123 Main St at 6 PM'")
    print("  - Policies: 'What are your opening hours?'")
    print("Type 'exit' or 'quit' to stop.\n")

    wat_tz = tz.gettz("Africa/Lagos")

    while True:
        user_input = input("How can I assist you? > ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the Restaurant Assistant. Goodbye!")
            break
        if not user_input:
            print("Please enter a valid request.")
            continue
        
        try:
            if "order" in user_input.lower() and "delivery" in user_input.lower():
                words = user_input.lower().split()
                address_start = words.index("to") if "to" in words else -1
                time_start = words.index("at") if "at" in words else -1
                if address_start == -1 or time_start == -1:
                    print("Please specify items, delivery address (after 'to'), and time (after 'at'). E.g., 'Order a Margherita Pizza for delivery to 123 Main St at 6 PM'.")
                    continue
                
                items = user_input[:user_input.lower().index(" for delivery")].replace("order ", "", 1).split(" and ")
                items = [item.strip() for item in items]
                delivery_address = " ".join(words[address_start + 1:time_start])
                delivery_time = " ".join(words[time_start + 1:])
                
                response = agent_executor.invoke({
                    "input": f"Order {', '.join(items)} for delivery to {delivery_address} at {delivery_time}"
                })
            elif "book" in user_input.lower() or "reserve" in user_input.lower():
                party_size=extract_party_size(user_input)

                if not party_size:
                    print(f"Could not detect party size in :'{user_input}")
                    print("Please specify a party size (e.g.,'Book a table for 4 at 7 PM').")
                    continue
                print(f"Detected prty size:{party_size}")

                try:
                    time_indicators=['at','tomorrow','today','tonight','pm','am','evening','morning','afternoon','night']
                    time_str=""

                    if "at" in user_input.lower():
                        time_str=user_input.lower().split("at",1)[1]

                    else:
                        words=user_input.split()
                        capturing=False
                        time_parts=[]
                        for word in words:
                            if any(indicator in word.lower()for indicator in time_indicators):
                                capturing=True

                            if capturing:
                                time_parts.append(word)

                        time_str="".join(time_parts)

                    if time_str.strip():
                        parsed_time=parse(time_str,fuzzy=True,default=datetime.datetime.now(tz=wat_tz))
                        print(f"Parsed time:{parsed_time.strftime('%I:%M %p on %B %d,%Y')}")

                    else:
                        parsed_time=datetime.datetime.now(tz=wat_tz)+ datetime.timedelta(hours=1)
                        print(f"No time specified. Using :{parsed_time.strftime('%I:%M %p')}")

                    formatted_time=parsed_time.isoformat()
                    response=agent_executor.invoke({
                        "input":f"Book a table for {party_size} people at {formatted_time}"
                    })

                except Exception as e:
                    print(f"Error parsing time:{e}")
                    print("Using default time(1 hour from now)....")
                    parsed_time=datetime.datetime.now(tz=wat_tz) + datetime.timedelta(hours=1)
                    formatted_time=parsed_time.isoformat()

                    response=agent_executor.invoke({
                        "input":f"Book a table for {party_size}people at {formatted_time}"
                    })
            else:
                response=agent_executor.invoke({"input":user_input})
            print("\nResponse:")
            print(response["output"])
            print()
        except Exception as e:
            print(f"Error processing request:{e}")
            traceback.print_exc()
            print()

if __name__=="__main__":
    main()


          

                
                        


          









