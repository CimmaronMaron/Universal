import requests
from requests.auth import AuthBase

class speed(url):
	def __init__(self):
		def glue(l):
			h=0
			for i in range(len(l)):
				h=h+l[i]
			return l
		from time import time
		start=time()
		requests.get(url)
		d1=time()-start
		requests.post(url)
		d2=time()-d1
		requests.patch(url)
		d3=time()-d2
		requests.request(url)
		d4=time()-d3
		k=[d1, d2, d3, d4]
		del(d1, d2, d3, d4)
		return glue(k)/len(k)
class TokenAuth(AuthBase):
	def __init__(self, token):
		self.token = token
	def __call__(self, r):
		r.headers['X-TokenAuth'] = f'{self.token}'  # Python 3.6+
		return r
