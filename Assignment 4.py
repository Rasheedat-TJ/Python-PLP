def calculate_discount(price, discount_percent):
    """
    Calculate the final price after applying a discount.
    Only apply discount if it's 20% or more.
    """
    if discount_percent >= 20:
        discount_amount = price * (discount_percent / 100)
        final_price = price - discount_amount
        return final_price
    else:
        return price

# Prompt user for inputs
try:
    price = float(input("Enter the original price: "))
    discount_percent = float(input("Enter the discount percentage: "))

    final_price = calculate_discount(price, discount_percent)

    if final_price < price:
        print(f"Discount applied! Final price: ${final_price:.2f}")
    else:
        print(f"No discount applied. Original price remains: ${price:.2f}")

except ValueError:
    print("Invalid input. Please enter numeric values.")
