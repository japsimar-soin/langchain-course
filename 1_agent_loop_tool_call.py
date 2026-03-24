from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

load_dotenv()


MAX_ITERATIONS = 15
MODEL = "qwen3:1.7b"
# MODEL = "gpt-5-nano"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of the product in the catalog."""

    print(f"Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 199.99, "keyboard": 129.99}

    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and retur the final price.
    Available tiers: 'gold', 'silver', 'bronze'"""

    print(f"Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"gold": 20, "silver": 12, "bronze": 7}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


@traceable(name="Langchain Agent Loop")
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a Product Catalog Tool "
                "and a Discount Tool.\n\n"
                "STRICT RULES - you must follow all these exactly as mentioned:\n"
                "1. NEVER guess or assume any product's price. "
                "You MUST call get_product_price() to get the real price of the product.\n"
                "2. You MUST call apply_discount() to apply the discount based on the"
                "given discount_tier after you have received the price using get_product_price().\n"
                "Pass the exact price returned by the get_product_price - DO NOT PASS a made-up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use - DO NOT ASSUME one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        print(f"AI Message: {ai_message.content}")
        tool_calls = ai_message.tool_calls
        print(f"Tool Calls: {tool_calls}")

        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}\n")
            return ai_message.content


        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected]: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool not found: {tool_name}")
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content = str(observation), tool_call_id = tool_call_id))

    print("ERROR: Exceeded MAX_ITERATIONS without a final answer.")
    return None


if __name__ == "__main__":
    print("hello agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop in gold tier?")
