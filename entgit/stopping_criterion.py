import numpy as np 


class StoppingCriterion:
	def __init__(self, n, m):
		self.n = n
		self.curr_m = 0
		self.val_history = []
		self.val_history_avg = [0]*m
		self.best_model_dict = None
		self.best_check = -1
		self.best_val = -1
		self.last_avg = -1
		self.checks = 0

	def check(self, val, model_dict):
		if val >= self.best_val:
			self.best_val = val
			self.best_model_dict = model_dict
			self.best_check = self.checks
		self.checks += 1
		self.val_history.append(val)
		if len(self.val_history) == self.n:
			curr_avg = np.mean(self.val_history)
			self.val_history_avg[self.curr_m] = curr_avg
			if self.curr_m == self.m - 1:
				self.curr_m = 0
			else:
				self.curr_m += 1

			if curr_avg < self.val_history_avg:
				return True
			else:
				self.last_avg = curr_avg
				self.val_history = []
		return False
	def reset(self):
		self.val_history = []
		self.best_model_dict = None
		self.best_check = -1
		self.best_val = -1
		self.last_avg = -1
		self.checks = 0

