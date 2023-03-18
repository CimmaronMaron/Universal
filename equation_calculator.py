class equation:
  def sqrt_eq(self, a, b, c):
    import math
    discriminant = b**2 - 4*a*c
		if discriminant > 0:
      x1 = (-b + math.sqrt(discriminant)) / (2*a)
      x2 = (-b - math.sqrt(discriminant)) / (2*a)
      return ("x1 = ", str(x1), "\nx2 = ", str(x2))
    elif discriminant == 0:
      x = -b / (2*a)
      return ("x = ", str(x))
    else:
      return (None)
	def ln_eq(a, b):
    if a == 0:
      if b == 0:
        return (0)
    else:
      return (None)
    else:
      x = -b / a
      return ("x = ", str(x))
