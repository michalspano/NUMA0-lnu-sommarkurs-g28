
# Task 1

class Interval:
    
  def __init__(self, left, right):
    if left >= right:
      self.left = right
      self.right = left
    else:
      self.left = left
      self.right = right
  
  def __repr__(self):
     return f"[{self.left, self.right}]"
# Task 3      
  def __str__(self):
     return f"[{self.left, self.right}]"
# Task 2  
  def __add__(self, other):
      return Interval(self.left + other.left, self.right + other.right)

  def __sub__(self, other):
      return Interval(self.left - other.right, self.right - other.left)

  def __mul__(self, other):
      products = [self.left * other.left, self.left * other.right, self.right * other.left, self.right * other.right]
      return Interval(min(products), max(products))

  def __truediv__(self, other):
      if other.left <= 0 <= other.right:
          raise ValueError("Division by an interval containing zero is undefined.")
      quotients = [self.left / other.left, self.left / other.right, self.right / other.left, self.right / other.right]
      return Interval(min(quotients), max(quotients))
 
  def __contains__(self, value):
      return self.left <= value <= self.right


# Task 4
# Test the arithmetic operations with example intervals
# I1 = Interval(1, 4)
# I2 = Interval(-2, -1)

# print(I1 + I2)
# print(I1 - I2)
# print(I1 * I2)
# print(I1 / I2)


# Task 5
# Test the __contains__ method
I1 = Interval(1, 4)

print(2 in I1)
print(0 in I1)
print(4 in I1)
print(5 in I1)

