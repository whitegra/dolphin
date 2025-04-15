import math

# ------------------ UTILITY FUNCTIONS ------------------

def zeros_like(data):
    return [[0 for _ in row] for row in data]

def shape_like(data):
    return (len(data), len(data[0])) if isinstance(data[0], list) else (len(data),)

def add_lists(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def sub_lists(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def mul_lists(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def div_lists(a, b):
    return [[a[i][j] / b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def sqrt_list(a):
    return [[math.sqrt(x) for x in row] for row in a]

def scalar_mul(a, scalar):
    return [[x * scalar for x in row] for row in a]

def scalar_div(a, scalar):
    return [[x / scalar for x in row] for row in a]

# ------------------ OPTIMIZERS ------------------

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                updates = scalar_mul(p.grad, self.lr)
                p.data = sub_lists(p.data, updates)

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = zeros_like(p.data)


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [zeros_like(p.data) for p in parameters]
        self.v = [zeros_like(p.data) for p in parameters]
        self.t = 1

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad and p.grad is not None:
                self.m[i] = add_lists(scalar_mul(self.m[i], self.beta1), scalar_mul(p.grad, 1 - self.beta1))
                self.v[i] = add_lists(scalar_mul(self.v[i], self.beta2), scalar_mul(mul_lists(p.grad, p.grad), 1 - self.beta2))

                m_hat = scalar_div(self.m[i], (1 - self.beta1 ** self.t))
                v_hat = scalar_div(self.v[i], (1 - self.beta2 ** self.t))

                updates = div_lists(m_hat, add_lists(sqrt_list(v_hat), [[self.eps]*len(v_hat[0])] * len(v_hat)))
                p.data = sub_lists(p.data, scalar_mul(updates, self.lr))

        self.t += 1

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = zeros_like(p.data)

class Momentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.requires_grad and p.grad is not None:
                self.velocity[i] = add_lists(scalar_mul(self.velocity[i], self.momentum), scalar_mul(p.grad, self.lr))
                p.data = sub_lists(p.data, self.velocity[i])

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = zeros_like(p.data)
