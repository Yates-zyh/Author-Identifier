from openai import OpenAI
import time
import random

# Initialize OpenAI client with API key and base URL
client = OpenAI(api_key="sk-5026edabe797479492b0ed7c9d8ad0ca", base_url="https://api.deepseek.com")

# Function to create a conversation with DeepSeek
def chat_with_deepseek():
    print("Welcome to DeepSeek chat! Type 'exit' to end the conversation.")

    # Dynamic system role input: Asking user to define the assistant's role
    system_role = input("Please define the assistant's role (e.g., 'helpful assistant', 'knowledgeable advisor','Creator of literary works' etc.): ")

    # Setting initial conversation context with dynamic system message
    conversation = [
        {"role": "system", "content": f"You are a {system_role}."}
    ]

    # Function to get user input with validation
    def get_user_input():
        while True:
            user_input = input("You: ")
            if user_input.strip():
                return user_input
            else:
                print("Input cannot be empty. Please enter a valid message.")

    while True:
        # Get user input with validation
        user_input = get_user_input()

        # Exit condition for the conversation
        if user_input.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            break

        # Add user input to the conversation history
        conversation.append({"role": "user", "content": user_input})

        # Attempt to call the API with retry mechanism
        attempt = 0
        while attempt < 3:
            try:
                # Fetch response from DeepSeek model using the conversation history
                print("Connecting to DeepSeek, please wait...")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=conversation,
                    stream=False
                )

                # Extract and print the response from the model
                deepseek_reply = response.choices[0].message.content
                print("DeepSeek:", deepseek_reply)

                # Add DeepSeek response to conversation history
                conversation.append({"role": "assistant", "content": deepseek_reply})
                break

            except Exception as e:
                # Print error message and retry
                print(f"Error occurred: {str(e)}. Retrying... ({attempt + 1}/3)")
                attempt += 1
                time.sleep(random.randint(1, 3))  # Random wait before retrying

        if attempt == 3:
            print("Failed to connect to DeepSeek after multiple attempts. Please try again later.")
            break

# Start the interactive chat
chat_with_deepseek()
