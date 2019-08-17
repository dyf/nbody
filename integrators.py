class Integrator:
    def __init__(self, derivative_fn):
        self.derivative_fn = derivative_fn
    
    @staticmethod
    def new(name, derivative_fn):
        for subclass in Integrator.__subclasses__():
            if subclass.__name__.lower() == name.lower():
                return subclass(derivative_fn)

    def step(self, dt, *X):
        pass

class Euler(Integrator):
    def step(self, dt, *X):
        return self.derivative_fn(dt, *X)

class RK2(Integrator):
    def step(self, dt, *X):
        dX = self.derivative_fn(dt, *X)
        return self.derivative_fn(dt*0.5, *(x + dx*dt*0.5 for x,dx in zip(X,dX)))

class RK4(Integrator):
    def step(self, dt, *X):
        dX1 = self.derivative_fn(dt, *X)
        dX2 = self.derivative_fn(dt*0.5, *(x + dx*dt*0.5 for x,dx in zip(X,dX1)))
        dX3 = self.derivative_fn(dt*0.5, *(x + dx*dt*0.5 for x,dx in zip(X,dX2)))
        dX4 = self.derivative_fn(dt, *(x + dx*dt*0.5 for x,dx in zip(X,dX3)))
        return ( (dx1 + 2*dx2 + 2*dx3 + dx4) / 6.0 for dx1,dx2,dx3,dx4 in zip(dX1,dX2,dX3,dX4) )


if __name__ == "__main__":
    print(Integrator.new('euler'))
        

