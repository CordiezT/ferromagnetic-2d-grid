import numpy as np
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, L, temperature):
        self.L = L
        self.temperature = temperature
        self.lattice = np.random.choice([-1, 1], size=(L, L))

    def metropolis_step(self):
        for _ in range(self.L**2):  # L^2 random spins
            i, j = np.random.randint(0, self.L, 2)
            spin = self.lattice[i, j]
            neighbours = self.lattice[(i+1)%self.L, j] + self.lattice[i, (j+1)%self.L] + \
                         self.lattice[(i-1)%self.L, j] + self.lattice[i, (j-1)%self.L]
            delta_E = 2 * spin * neighbours
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.temperature):
                self.lattice[i, j] *= -1

    def simulate(self, steps):
        for _ in range(steps):
            self.metropolis_step()
        return self.lattice

def magnetisation(lattice):
    return np.sum(lattice) / lattice.size

def main():
    L = 50  # Lattice size
    temperatures = np.linspace(1.5, 3.5, 20)
    steps = 1000

    magnetisations = []

    for T in temperatures:
        model = IsingModel(L, T)
        lattice = model.simulate(steps)
        mag = magnetisation(lattice)
        magnetisations.append(mag)

        plt.imshow(lattice, cmap='coolwarm', interpolation='nearest')
        plt.title(f'Temperature = {T:.2f}')
        plt.show()

    plt.plot(temperatures, magnetisations, 'o-')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetisation')
    plt.title('Magnetisation vs Temperature')
    plt.show()

if __name__ == "__main__":
    main()
