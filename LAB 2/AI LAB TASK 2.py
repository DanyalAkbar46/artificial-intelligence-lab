last_value = 0

for number in range(1, 101):

    total = last_value + number

    # Decide correct answer based on TOTAL
    if total % 3 == 0 and total % 5 == 0:
        correct_answer = "Fizz Buzz"
    elif total % 3 == 0:
        correct_answer = "Fizz"
    elif total % 5 == 0:
        correct_answer = "Buzz"
    else:
        correct_answer = "None"

    # Display only the new number
    user_answer = input(f"Number is {number}. Your answer: ")

    if user_answer == correct_answer:
        print("Correct ✅")
    else:
        print("Wrong ❌")
        print("Correct answer was:", correct_answer)
        break

    # Store current number for next round
    last_value = number

