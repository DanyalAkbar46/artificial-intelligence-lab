def luhn_algorithm(card_number: str) -> bool:
   
    digits = [int(d) for d in card_number[::-1]]
    
   
    for i in range(1, len(digits), 2):
        doubled = digits[i] * 2
       
        digits[i] = doubled - 9 if doubled > 9 else doubled
    
   
    total = sum(digits)
    
   
    return total % 10 == 0


card = "4532015112830366"
print("Valid" if luhn_algorithm(card) else "Invalid")