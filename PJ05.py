import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
import matplotlib
from sympy import MatrixSymbol, Matrix
from sympy import *
import copy

class PJ05:
	def __init__(self, dt=0.02, sigma=0.2, max_iteration=400):
		self.dt = dt
		self.sigma = sigma
		self.max_iteration = max_iteration

	def reset(self, x=None, y=None):
		if(x == None and y == None):
			x = np.random.uniform(low=-2.4, high=-1.6)
			y = np.random.uniform(low=-0.4, high=0.4)
			while((x + 2)**2 + y**2 > 0.1):
				x = np.random.uniform(low=-2.4, high=-1.6)
				y = np.random.uniform(low=-0.4, high=0.4)
		self.x = x
		self.y = y
		self.state = np.array([self.x, self.y])
		self.t = 0

		return self.state

	def step(self, u):
		x, y = self.x, self.y
		x += self.dt*y
		y += self.dt*(u - 0.5*x**3) + self.sigma*np.random.normal(0, np.sqrt(self.dt))
		self.x, self.y = x, y
		self.state = np.array([self.x, self.y])
		self.t += 1
		done = self.t == self.max_iteration
		return self.state, 0, done


def generateConstraints(x, y, exp1, exp2, degree):
	constraints = []
	for i in range(degree+1):
		for j in range(degree+1):
			if exp1.coeff(x, i).coeff(y, j) != 0:
				print('constraints += [', exp1.coeff(x, i).coeff(y, j), ' == ', exp2.coeff(x, i).coeff(y, j), ']')


def BarrierSDP(control_param, s, weight):
	M = cp.Variable((6, 6), symmetric=True)
	N = cp.Variable((6, 6), symmetric=True)
	J = cp.Variable((6, 6), symmetric=True)
	K = cp.Variable((10, 10), symmetric=True)
	A = cp.Variable((3, 3), symmetric=True)
	C = cp.Variable((3, 3), symmetric=True)
	D = cp.Variable((3, 3), symmetric=True)
	E = cp.Variable((3, 3), symmetric=True)
	a = cp.Variable((1,6))
	c = cp.Variable((1,6))
	d = cp.Variable((1,6))
	e = cp.Variable((1,6))
	Pr = cp.Variable()
	B = cp.Variable((1, 15))
	t = cp.Parameter((1, 2))
	slack = cp.Variable()

	objective = cp.Minimize(Pr**2 + weight*slack**2)

	constraints = []
	constraints += [ slack >= 0]
	constraints += [ Pr >= 0 ]
	constraints += [ M >> 0 ]
	constraints += [ N >> 0 ]
	constraints += [ J >> 0 ]
	constraints += [ K >> 0 ]
	constraints += [ A >> 0 ]
	constraints += [ C >> 0 ]
	constraints += [ D >> 0 ]
	constraints += [ E >> 0 ]	

	constraints += [ M[0, 0]  >=  B[0, 0] - 16*a[0, 0] - slack]
	constraints += [ M[0, 0]  <=  B[0, 0] - 16*a[0, 0] + slack]
	constraints += [ M[0, 2] + M[2, 0]  ==  B[0, 2] - 16*a[0, 2] ]
	constraints += [ M[0, 5] + M[2, 2] + M[5, 0]  ==  B[0, 5] + a[0, 0] - 16*a[0, 5] ]
	constraints += [ M[2, 5] + M[5, 2]  ==  B[0, 9] + a[0, 2] ]
	constraints += [ M[5, 5]  ==  B[0, 14] + a[0, 5] ]
	constraints += [ M[0, 1] + M[1, 0]  ==  B[0, 1] - 16*a[0, 1] ]
	constraints += [ M[0, 4] + M[1, 2] + M[2, 1] + M[4, 0]  ==  B[0, 3] - 16*a[0, 3] ]
	constraints += [ M[1, 5] + M[2, 4] + M[4, 2] + M[5, 1]  ==  B[0, 8] + a[0, 1] ]
	constraints += [ M[4, 5] + M[5, 4]  ==  B[0, 13] + a[0, 3] ]
	constraints += [ M[0, 3] + M[1, 1] + M[3, 0]  ==  B[0, 4] + a[0, 0] - 16*a[0, 4] ]
	constraints += [ M[1, 4] + M[2, 3] + M[3, 2] + M[4, 1]  ==  B[0, 7] + a[0, 2] ]
	constraints += [ M[3, 5] + M[4, 4] + M[5, 3]  ==  B[0, 12] + a[0, 4] + a[0, 5] ]
	constraints += [ M[1, 3] + M[3, 1]  ==  B[0, 6] + a[0, 1] ]
	constraints += [ M[3, 4] + M[4, 3]  ==  B[0, 11] + a[0, 3] ]
	constraints += [ M[3, 3]  ==  B[0, 10] + a[0, 4] ]
	constraints += [ N[0, 0]  ==  B[0, 0] + 2.25*c[0, 0] - 1 ]
	constraints += [ N[0, 2] + N[2, 0]  ==  B[0, 2] - c[0, 0] + 2.25*c[0, 2] ]
	constraints += [ N[0, 5] + N[2, 2] + N[5, 0]  ==  B[0, 5] - c[0, 2] + 2.25*c[0, 5] ]
	constraints += [ N[2, 5] + N[5, 2]  ==  B[0, 9] - c[0, 5] ]
	constraints += [ N[5, 5]  ==  B[0, 14] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  B[0, 1] + 2.25*c[0, 1] ]
	constraints += [ N[0, 4] + N[1, 2] + N[2, 1] + N[4, 0]  ==  B[0, 3] - c[0, 1] + 2.25*c[0, 3] ]
	constraints += [ N[1, 5] + N[2, 4] + N[4, 2] + N[5, 1]  ==  B[0, 8] - c[0, 3] ]
	constraints += [ N[4, 5] + N[5, 4]  ==  B[0, 13] ]
	constraints += [ N[0, 3] + N[1, 1] + N[3, 0]  ==  B[0, 4] + 2.25*c[0, 4] ]
	constraints += [ N[1, 4] + N[2, 3] + N[3, 2] + N[4, 1]  ==  B[0, 7] - c[0, 4] ]
	constraints += [ N[3, 5] + N[4, 4] + N[5, 3]  ==  B[0, 12] ]
	constraints += [ N[1, 3] + N[3, 1]  ==  B[0, 6] ]
	constraints += [ N[3, 4] + N[4, 3]  ==  B[0, 11] ]
	constraints += [ N[3, 3]  ==  B[0, 10] ]
	constraints += [ J[0, 0]  >=  Pr - B[0, 0] + 3.9*d[0, 0] - slack]
	constraints += [ J[0, 0]  <=  Pr - B[0, 0] + 3.9*d[0, 0] + slack]
	constraints += [ J[0, 2] + J[2, 0]  ==  -B[0, 2] + 3.9*d[0, 2] ]
	constraints += [ J[0, 5] + J[2, 2] + J[5, 0]  ==  -B[0, 5] + d[0, 0] + 3.9*d[0, 5] ]
	constraints += [ J[2, 5] + J[5, 2]  ==  -B[0, 9] + d[0, 2] ]
	constraints += [ J[5, 5]  ==  -B[0, 14] + d[0, 5] ]
	constraints += [ J[0, 1] + J[1, 0]  ==  -B[0, 1] + 4*d[0, 0] + 3.9*d[0, 1] ]
	constraints += [ J[0, 4] + J[1, 2] + J[2, 1] + J[4, 0]  ==  -B[0, 3] + 4*d[0, 2] + 3.9*d[0, 3] ]
	constraints += [ J[1, 5] + J[2, 4] + J[4, 2] + J[5, 1]  ==  -B[0, 8] + d[0, 1] + 4*d[0, 5] ]
	constraints += [ J[4, 5] + J[5, 4]  ==  -B[0, 13] + d[0, 3] ]
	constraints += [ J[0, 3] + J[1, 1] + J[3, 0]  ==  -B[0, 4] + d[0, 0] + 4*d[0, 1] + 3.9*d[0, 4] ]
	constraints += [ J[1, 4] + J[2, 3] + J[3, 2] + J[4, 1]  ==  -B[0, 7] + d[0, 2] + 4*d[0, 3] ]
	constraints += [ J[3, 5] + J[4, 4] + J[5, 3]  ==  -B[0, 12] + d[0, 4] + d[0, 5] ]
	constraints += [ J[1, 3] + J[3, 1]  ==  -B[0, 6] + d[0, 1] + 4*d[0, 4] ]
	constraints += [ J[3, 4] + J[4, 3]  ==  -B[0, 11] + d[0, 3] ]
	constraints += [ J[3, 3]  ==  -B[0, 10] + d[0, 4] ]
	constraints += [ K[0, 0]  ==  -1.0*s**2*B[0, 4] - 1.0*s**2*B[0, 5] - 16*e[0, 0] ]
	constraints += [ K[0, 2] + K[2, 0]  ==  -1.0*s**2*B[0, 7] - 3.0*s**2*B[0, 9] - B[0, 1] - B[0, 2]*t[0, 1] - 16*e[0, 2] ]
	constraints += [ K[0, 5] + K[2, 2] + K[5, 0]  ==  -1.0*s**2*B[0, 12] - 6.0*s**2*B[0, 14] - B[0, 3] - 2*B[0, 5]*t[0, 1] + e[0, 0] - 16*e[0, 5] ]
	constraints += [ K[0, 9] + K[2, 5] + K[5, 2] + K[9, 0]  ==  -B[0, 8] - 3*B[0, 9]*t[0, 1] + e[0, 2] ]
	constraints += [ K[2, 9] + K[5, 5] + K[9, 2]  ==  -B[0, 13] - 4*B[0, 14]*t[0, 1] + e[0, 5] ]
	constraints += [ K[5, 9] + K[9, 5]  ==  0 ]
	constraints += [ K[9, 9]  ==  0 ]
	constraints += [ K[0, 1] + K[1, 0]  ==  -3.0*s**2*B[0, 6] - 1.0*s**2*B[0, 8] - B[0, 2]*t[0, 0] - 16*e[0, 1] ]
	constraints += [ K[0, 3] + K[1, 2] + K[2, 1] + K[3, 0]  ==  -3.0*s**2*B[0, 11] - 3.0*s**2*B[0, 13] - B[0, 3]*t[0, 1] - 2*B[0, 4] - 2*B[0, 5]*t[0, 0] - 16*e[0, 3] ]
	constraints += [ K[0, 8] + K[1, 5] + K[2, 3] + K[3, 2] + K[5, 1] + K[8, 0]  ==  -2*B[0, 7] - 2*B[0, 8]*t[0, 1] - 3*B[0, 9]*t[0, 0] + e[0, 1] ]
	constraints += [ K[1, 9] + K[2, 8] + K[3, 5] + K[5, 3] + K[8, 2] + K[9, 1]  ==  -2*B[0, 12] - 3*B[0, 13]*t[0, 1] - 4*B[0, 14]*t[0, 0] + e[0, 3] ]
	constraints += [ K[3, 9] + K[5, 8] + K[8, 5] + K[9, 3]  ==  0 ]
	constraints += [ K[8, 9] + K[9, 8]  ==  0 ]
	constraints += [ K[0, 4] + K[1, 1] + K[4, 0]  ==  -6.0*s**2*B[0, 10] - 1.0*s**2*B[0, 12] - B[0, 3]*t[0, 0] + e[0, 0] - 16*e[0, 4] ]
	constraints += [ K[0, 7] + K[1, 3] + K[2, 4] + K[3, 1] + K[4, 2] + K[7, 0]  ==  -3*B[0, 6] - B[0, 7]*t[0, 1] - 2*B[0, 8]*t[0, 0] + e[0, 2] ]
	constraints += [ K[1, 8] + K[2, 7] + K[3, 3] + K[4, 5] + K[5, 4] + K[7, 2] + K[8, 1]  ==  -3*B[0, 11] - 2*B[0, 12]*t[0, 1] - 3*B[0, 13]*t[0, 0] + e[0, 4] + e[0, 5] ]
	constraints += [ K[3, 8] + K[4, 9] + K[5, 7] + K[7, 5] + K[8, 3] + K[9, 4]  ==  0 ]
	constraints += [ K[7, 9] + K[8, 8] + K[9, 7]  ==  0 ]
	constraints += [ K[0, 6] + K[1, 4] + K[4, 1] + K[6, 0]  ==  0.5*B[0, 2] - B[0, 7]*t[0, 0] + e[0, 1] ]
	constraints += [ K[1, 7] + K[2, 6] + K[3, 4] + K[4, 3] + K[6, 2] + K[7, 1]  ==  1.0*B[0, 5] - 4*B[0, 10] - B[0, 11]*t[0, 1] - 2*B[0, 12]*t[0, 0] + e[0, 3] ]
	constraints += [ K[3, 7] + K[4, 8] + K[5, 6] + K[6, 5] + K[7, 3] + K[8, 4]  ==  1.5*B[0, 9] ]
	constraints += [ K[6, 9] + K[7, 8] + K[8, 7] + K[9, 6]  ==  2.0*B[0, 14] ]
	constraints += [ K[1, 6] + K[4, 4] + K[6, 1]  ==  0.5*B[0, 3] - B[0, 11]*t[0, 0] + e[0, 4] ]
	constraints += [ K[3, 6] + K[4, 7] + K[6, 3] + K[7, 4]  ==  1.0*B[0, 8] ]
	constraints += [ K[6, 8] + K[7, 7] + K[8, 6]  ==  1.5*B[0, 13] ]
	constraints += [ K[4, 6] + K[6, 4]  ==  0.5*B[0, 7] ]
	constraints += [ K[6, 7] + K[7, 6]  ==  1.0*B[0, 12] ]
	constraints += [ K[6, 6]  ==  0.5*B[0, 11] ]
	constraints += [ A[0, 0]  ==  a[0, 0] ]
	constraints += [ A[0, 2] + A[2, 0]  ==  a[0, 2] ]
	constraints += [ A[2, 2]  ==  a[0, 5] ]
	constraints += [ A[0, 1] + A[1, 0]  ==  a[0, 1] ]
	constraints += [ A[1, 2] + A[2, 1]  ==  a[0, 3] ]
	constraints += [ A[1, 1]  ==  a[0, 4] ]
	constraints += [ C[0, 0]  ==  c[0, 0] ]
	constraints += [ C[0, 2] + C[2, 0]  ==  c[0, 2] ]
	constraints += [ C[2, 2]  ==  c[0, 5] ]
	constraints += [ C[0, 1] + C[1, 0]  ==  c[0, 1] ]
	constraints += [ C[1, 2] + C[2, 1]  ==  c[0, 3] ]
	constraints += [ C[1, 1]  ==  c[0, 4] ]
	constraints += [ D[0, 0]  >=  d[0, 0] - slack ]
	constraints += [ D[0, 0]  <=  d[0, 0] + slack ]
	constraints += [ D[0, 2] + D[2, 0]  ==  d[0, 2] ]
	constraints += [ D[2, 2]  ==  d[0, 5] ]
	constraints += [ D[0, 1] + D[1, 0]  ==  d[0, 1] ]
	constraints += [ D[1, 2] + D[2, 1]  ==  d[0, 3] ]
	constraints += [ D[1, 1]  ==  d[0, 4] ]
	constraints += [ E[0, 0]  ==  e[0, 0] ]
	constraints += [ E[0, 2] + E[2, 0]  ==  e[0, 2] ]
	constraints += [ E[2, 2]  ==  e[0, 5] ]
	constraints += [ E[0, 1] + E[1, 0]  ==  e[0, 1] ]
	constraints += [ E[1, 2] + E[2, 1]  ==  e[0, 3] ]
	constraints += [ E[1, 1]  ==  e[0, 4] ]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[a,c,d,e, Pr, slack, M,N,J,K,B, A,C,D,E])
	a_star,c_star,d_star,e_star, Pr_star, slack_star, M_star,N_star,J_star,K_star,B_star,_,_,_,_ = layer(theta_t)
	Pr_star.backward()
	# print(theta_t.grad.detach().numpy()[0])
	Pr_grad = copy.deepcopy(theta_t.grad.detach().numpy()[0])
	slack_star.backward()
	# print(theta_t.grad.detach().numpy()[0])
	slack_grad = theta_t.grad.detach().numpy()[0]

	return B_star.detach().numpy(), Pr_star.detach().numpy(), slack_star.detach().numpy(), Pr_grad, slack_grad

def BarrierConstraint():
	x, y, Pr, s = symbols('x, y,  Pr, s')
	Bbase = Matrix([1, x, y, x*y, x**2, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4])
	B = MatrixSymbol('B', 1, 15)
	M = MatrixSymbol('M', 6, 6)
	N = MatrixSymbol('N', 6, 6) 
	J = MatrixSymbol('J', 6, 6)
	K = MatrixSymbol('K', 10, 10)

	A = MatrixSymbol('A', 3, 3)
	a = MatrixSymbol('a', 1, 6)
	C = MatrixSymbol('C', 3, 3)
	c = MatrixSymbol('c', 1, 6)
	D = MatrixSymbol('D', 3, 3)
	d = MatrixSymbol('d', 1, 6)
	E = MatrixSymbol('E', 3, 3)
	e = MatrixSymbol('e', 1, 6)

	# B(x) >= 0 for X
	MatrixBase = Matrix([1, x, y, x**2, x*y, y**2])
	rhsM = MatrixBase.T*M*MatrixBase
	rhsM = expand(rhsM[0, 0])

	lhsM = B*Bbase - a*Matrix([1,x,y,x*y,x**2, y**2])*Matrix([16 - x**2 - y**2])
	lhsM = expand(lhsM[0, 0])
	generateConstraints(x, y, rhsM, lhsM, degree=4)


	# B(x) >= 1 for Xu
	rhsN = MatrixBase.T*N*MatrixBase
	rhsN = expand(rhsN[0, 0])
	lhsN = B*Bbase - Matrix([1]) - c*Matrix([1,x,y,x*y,x**2, y**2])*Matrix([y - 2.25])
	lhsN = expand(lhsN[0, 0])
	generateConstraints(x, y, rhsN, lhsN, degree=4)


	# B(x) <= gamma for X0
	rhsJ = MatrixBase.T*J*MatrixBase
	rhsJ = expand(rhsJ[0, 0])
	lhsJ = Matrix([Pr]) - B*Bbase - d*Matrix([1,x,y,x*y,x**2, y**2])*Matrix([0.1 - (x+2)**2 - y**2])
	lhsJ = expand(lhsJ[0, 0])
	generateConstraints(x, y, rhsJ, lhsJ, degree=4)


	# lie derivative for X
	rhsK = Matrix([1, x, y, x*y, x**2, y**2, x**3, x**2*y, x*y**2, y**3]).T*K*Matrix([1, x, y, x*y, x**2, y**2, x**3, x**2*y, x*y**2, y**3])
	rhsK = expand(rhsK[0, 0])

	Barrier = B*Bbase
	Barrier = expand(Barrier[0, 0])
	partialx = diff(Barrier, x)
	partialy = diff(Barrier, y)
	partialxdouble = diff(partialx, x)
	partialydouble = diff(partialy, y)

	theta = MatrixSymbol('t', 1, 2)
	controlInput = theta*Matrix([[x], [y]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[y], [controlInput-0.5*x**3]])
	lhsK = -Matrix([[partialx], [partialy]]).T*dynamics - 0.5*s**2*Matrix([partialxdouble + partialydouble]) - e*Matrix([1,x,y,x*y,x**2, y**2])*Matrix([16 - x**2 - y**2])
	lhsK = expand(lhsK[0, 0])
	generateConstraints(x, y, rhsK, lhsK, degree=6)

	a_SOS_right = Matrix([1,x,y]).T*A*Matrix([1,x,y])
	a_SOS_right = expand(a_SOS_right[0, 0])
	a_SOS_left = a*Matrix([1,x,y,x*y,x**2, y**2])
	a_SOS_left = expand(a_SOS_left[0, 0])
	generateConstraints(x,y, a_SOS_right, a_SOS_left, degree=2)

	c_SOS_right = Matrix([1,x,y]).T*C*Matrix([1,x,y])
	c_SOS_right = expand(c_SOS_right[0, 0])
	c_SOS_left = c*Matrix([1,x,y,x*y,x**2, y**2])
	c_SOS_left = expand(c_SOS_left[0, 0])
	generateConstraints(x,y, c_SOS_right, c_SOS_left, degree=2)

	d_SOS_right = Matrix([1,x,y]).T*D*Matrix([1,x,y])
	d_SOS_right = expand(d_SOS_right[0, 0])
	d_SOS_left = d*Matrix([1,x,y,x*y,x**2, y**2])
	d_SOS_left = expand(d_SOS_left[0, 0])
	generateConstraints(x,y, d_SOS_right, d_SOS_left, degree=2)

	e_SOS_right = Matrix([1,x,y]).T*E*Matrix([1,x,y])
	e_SOS_right = expand(e_SOS_right[0, 0])
	e_SOS_left = e*Matrix([1,x,y,x*y,x**2, y**2])
	e_SOS_left = expand(e_SOS_left[0, 0])
	generateConstraints(x,y, e_SOS_right, e_SOS_left, degree=2)


def plot(Barrier, control_param, Prob):
	env = PJ05(sigma=0.1)
	state, done = env.reset(), False
	trajectory = []
	while not done:
		u = state.dot(control_param)
		trajectory.append(state)
		state, _, done = env.step(u)

	trajectory = np.array(trajectory)
	fig = plt.figure(0)
	ax = fig.add_subplot(111)
	plt.plot(trajectory[:, 0], trajectory[:, 1])
	# print(Barrier[0].shape)

	x = np.linspace(-4, 4, 50)
	y = np.linspace(-4, 4, 50)
	x,y = np.meshgrid(x, y)
	z = Barrier[0].dot(np.array([1, x, y, x*y, x**2, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4], dtype=object))
	levels = np.array([Prob,1])
	cs = plt.contour(x, y, z, levels)
	rect1 = matplotlib.patches.Rectangle((-4,2.25), 8, 1.75, color='red')
	ax.add_patch(rect1)
	plt.xlim([-2.5, -1])
	plt.ylim([-1, 1])
	plt.savefig('Tra_Barrier_PJ_detailed.png')
	


if __name__ == '__main__':
	# env = PJ05(sigma=0.1)
	# state, done = env.reset(), False
	# trajectory = []
	# while not done:
	# 	u = 2 * state[0] - 3.9*state[1]
	# 	trajectory.append(state)
	# 	state, _, done = env.step(u)
	# 	print(state)

	# trajectory = np.array(trajectory)
	# plt.plot(trajectory[:, 0], trajectory[:, 1])
	# plt.savefig('tra.png')
	# assert False

	# BarrierConstraint()
	# assert False

	# control_param = np.array([2, -3.9])
	# Barrier, Prob, slack, Pr_Derivative, slack_Derivative = BarrierSDP(control_param, s=0.1, weight=100)
	# print(Barrier, Prob, slack, Pr_Derivative, slack_Derivative)	
	# plot(Barrier, control_param, Prob)
	# assert False

	control_param = np.array([0.0, 0.0])
	weights = np.linspace(0.1, 100, 100)
	for i in range(100):
		Barrier, Prob, slack, Pr_Derivative, slack_Derivative = BarrierSDP(control_param, s=0.1, weight=weights[i])
		control_param -= 10*slack_Derivative
		control_param -= 10*Pr_Derivative
		print(control_param, Prob, slack, Pr_Derivative, slack_Derivative)
