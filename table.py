class table():
  def __init__(self, Column1, Column1i, value, value2='', Column2='', Column2i=''):
    self.Column1=Column1
    self.Column1i=Column1i
    self.Column2=Column2
    self.Column2i=Column2i
    k=''
    self.k=k
    i = f'''
     {self.k}{Column1}
    1{self.k}{Column1i}{k}{value}
    2{self.k}{Column2i}{k}{value2}
    '''
    return i
