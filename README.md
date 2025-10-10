 🍽️ NEXUS Restaurant AI Assistant

An intelligent restaurant management assistant powered by LangChain, OpenAI GPT, and FAISS.
This assistant can handle:

 Table reservations (with simulated POS & Google Calendar sync)

 Food delivery orders

 Menu and restaurant policy inquiries

 Semantic search across menu items and FAQs using FAISS embeddings





 Features

 Core Capabilities

Menu Queries: Retrieve menu items, categories, prices, and descriptions.

Reservations: Automatically check table availability, create bookings, and simulate POS + reminder systems.

Food Delivery: Validate items, schedule deliveries, and simulate integration with a delivery system.

Policies & FAQs: Access predefined policies like hours, reservations, dietary options, etc…
 
├── bookings.json              # Local database for bookings & orders
├── restaurantmenuchanges.cs   # Menu dataset (gotten from kaggle)
├── faiss_restaurant_index/    # Stored FAISS embeddings
├── .env                       # Environment variables (contains OPENAI_API_KEY)
├── main.py                    # Main application script
└── README.md                  # Project documentation


Clone the Repository

git clone https://github.com/yourusername/restaurant-assistant.git
cd restaurant-assistant

Create a Virtual Environment

python -m venv myenv
source venv/bin/activate     # On Mac/Linux
myenv\Scripts\activate        # On Windows

Install Dependencies

pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install langchain langchain-openai langchain-community pandas faiss-cpu python-dotenv python-dateutil openai

Add Your OpenAI API Key

Create a .env file in your project root:

OPENAI_API_KEY=your_openai_api_key_here

 How It Works

1. Initialization

Loads menu data from restaurantmenuchanges.csv

Builds or loads a FAISS vector index

Creates a local bookings.json database



2. Conversation Flow

User interacts through the terminal.

The agent decides which tool to call (reservation, menu, delivery, etc.).

Responses are generated via OpenAI GPT-4o, using contextual memory.



3. Reservation Workflow

check_table_availability → book_google_calendar_event → reserve_table_locally


4. Delivery Workflow

get_menu_items → create_delivery_order → mock_delivery_system




Example Interactions

Menu Query:

> What vegan options do you have?



Reservation:

> Book a table for 4 at 7 PM tomorrow.



Delivery:

> Order a Margherita Pizza and Caesar Salad for delivery to 123 Main St at 6 PM.



Policy:

> What are your opening hours?



 Data Files

bookings.json – stores:

{
  "tables": {"T1": {"capacity": 2, "status": "available"}},
  "bookings": [],
  "orders": []
}

faiss_restaurant_index/ – local FAISS vector index storing menu embeddings for fast semantic search.




Running the App

Start the assistant:

python main.py

Then interact via terminal:

🍽️ Welcome to the Restaurant Assistant! 🍽️
How can I assist you? > Book a table for 2 at 6 PM today

To exit:

How can I assist you? > exit




Tools Used by the Agent


 Memory & Context

Uses ConversationBufferMemory to maintain chat history.

Keeps context between user and assistant for a natural flow.



 Timezone

All date/time operations are handled in Africa/Lagos timezone using:

from dateutil import tz
tz.gettz("Africa/Lagos")





