# pip install google-generativeai mcp crewai crewai-tools
import asyncio
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Gemini LLM for CrewAI
gemini_llm = LLM(
    model="gemini/gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY")
)

# MCP Server parameters
server_params = StdioServerParameters(
    command="mcp-flight-search",
    args=["--connection_type", "stdio"],
    env={"SERP_API_KEY": os.getenv("SERP_API_KEY")},
)

# Create a custom CrewAI tool that wraps MCP functionality
@tool("Flight Search Tool")
def flight_search_tool(origin: str, destination: str, date: str) -> str:
    """
    Search for flights between two cities on a specific date.
    
    Args:
        origin: Departure city
        destination: Arrival city
        date: Travel date in YYYY-MM-DD format
    
    Returns:
        Flight search results in JSON format
    """
    # This will be populated by the async MCP call
    return f"Searching flights from {origin} to {destination} on {date}"


async def get_mcp_flight_data(origin: str, destination: str, date: str):
    """Async function to interact with MCP server"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            mcp_tools = await session.list_tools()
            
            # Convert MCP tools to Gemini format
            tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in mcp_tools.tools
            ]
            
            prompt = f"Find Flights from {origin} to {destination} {date}"
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=tools,
                ),
            )
            
            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                
                result = await session.call_tool(
                    function_call.name, arguments=dict(function_call.args)
                )
                
                try:
                    flight_data = json.loads(result.content[0].text)
                    return json.dumps(flight_data, indent=2)
                except json.JSONDecodeError:
                    return result.content[0].text
            else:
                return "No flights found or function call was not generated."


async def run_crewai_with_mcp():
    """Main function to run CrewAI agents with MCP integration"""
    
    # Flight parameters
    origin = "Atlanta"
    destination = "Las Vegas"
    date = "2025-12-12"
    
    # Get flight data from MCP
    print("🔍 Fetching flight data via MCP...")
    flight_data = await get_mcp_flight_data(origin, destination, date)
    print("\n--- Flight Data Retrieved ---")
    print(flight_data)
    print("\n")
    
    # Define CrewAI Agents
    flight_researcher = Agent(
        role='Flight Research Specialist',
        goal=f'Analyze flight options from {origin} to {destination} on {date} and identify the best options',
        backstory='You are an expert in analyzing flight data, comparing prices, durations, and finding the best deals for travelers.',
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )
    
    travel_advisor = Agent(
        role='Travel Advisor',
        goal='Provide comprehensive travel recommendations based on flight analysis',
        backstory='You are a seasoned travel advisor who helps clients make informed decisions about their trips, considering factors like price, convenience, and comfort.',
        verbose=True,
        allow_delegation=True,
        llm=gemini_llm
    )
    
    # Define Tasks
    analyze_flights_task = Task(
        description=f"""
        Analyze the following flight data and identify:
        1. The cheapest flight option
        2. The fastest flight option
        3. The best overall value considering price and duration
        
        Flight Data:
        {flight_data}
        """,
        agent=flight_researcher,
        expected_output='A detailed analysis of flight options with clear recommendations for cheapest, fastest, and best value flights.'
    )
    
    create_recommendation_task = Task(
        description="""
        Based on the flight analysis, create a comprehensive travel recommendation that includes:
        1. Top 3 flight recommendations with reasoning
        2. Pros and cons of each option
        3. Final recommendation for different traveler types (budget, business, balanced)
        """,
        agent=travel_advisor,
        expected_output='A well-structured travel recommendation report with clear guidance for different traveler preferences.',
        context=[analyze_flights_task]
    )
    
    # Create and run the crew
    print("🤖 Starting CrewAI agents...")
    crew = Crew(
        agents=[flight_researcher, travel_advisor],
        tasks=[analyze_flights_task, create_recommendation_task],
        verbose=True
    )
    
    result = crew.kickoff()
    
    print("\n" + "="*50)
    print("📋 FINAL CREW OUTPUT")
    print("="*50)
    print(result)
    
    return result


# Alternative: Simple example without MCP (for testing CrewAI + Gemini)
async def run_simple_crewai_example():
    """Simple CrewAI example with Gemini LLM"""
    
    researcher = Agent(
        role='AI Researcher',
        goal='Research and explain artificial intelligence concepts',
        backstory='You are an AI expert with deep knowledge of machine learning and natural language processing.',
        verbose=True,
        llm=gemini_llm
    )
    
    writer = Agent(
        role='Content Writer',
        goal='Create clear and engaging explanations of technical concepts',
        backstory='You are a skilled technical writer who can make complex topics accessible.',
        verbose=True,
        llm=gemini_llm
    )
    
    research_task = Task(
        description='Research and compile key information about Natural Language Processing (NLP)',
        agent=researcher,
        expected_output='A comprehensive overview of NLP including key concepts, techniques, and applications.'
    )
    
    writing_task = Task(
        description='Write a clear, engaging explanation of NLP for a general audience',
        agent=writer,
        expected_output='A well-written article about NLP that is accessible to non-technical readers.',
        context=[research_task]
    )
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n" + "="*50)
    print("📋 FINAL OUTPUT")
    print("="*50)
    print(result)


# Main execution
if __name__ == "__main__":
    print("Choose mode:")
    print("1. Full integration (CrewAI + MCP + Gemini)")
    print("2. Simple CrewAI + Gemini example")
    
    # For full integration with MCP
    asyncio.run(run_crewai_with_mcp())
    
    # OR for simple example without MCP
    # asyncio.run(run_simple_crewai_example())