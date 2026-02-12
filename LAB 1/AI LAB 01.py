def dynamic_calculator(expression):
    try:
        result = eval(expression)
        return result
    except:
        return "Invalid Expression"


while True:
    print("\nDynamic Calculator")
    print("Example: 2+3*5-4/2")

    exp = input("Enter your equation: ")
    answer = dynamic_calculator(exp)
    print("Answer =", answer)

    choice = input("Do you want to calculate again? (yes/no): ")
    if choice.lower() != "yes":
        print("Calculator Closed")
        break

