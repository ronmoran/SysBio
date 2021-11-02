import numpy as np
import pandas
import plotly.express as px


class SIRModel:

    def __init__(self,
                 infection_const: float,
                 recovery_const: float,
                 vaccination_const: float,
                 vaccination_decay_const: float,
                 vaccinated_infection_const: float,
                 dt: float):
        self.__infection_const = infection_const
        self.__recovery_const = recovery_const
        self.__vaccination_const = vaccination_const
        self.__vaccination_decay_const = vaccination_decay_const
        self.__vaccinated_infection_const = vaccinated_infection_const
        self.__steps = int(100/dt)
        self.__susceptible = np.empty(self.__steps)
        self.__vaccinated = np.empty(self.__steps)
        self.__infected = np.empty(self.__steps)
        self.__recovered = np.empty(self.__steps)
        self.__dt = dt

    @property
    def infected(self):
        return self.__infected

    @property
    def vaccinated(self):
        return self.__vaccinated

    @property
    def recovered(self):
        return self.__recovered

    @property
    def susceptible(self):
        return self.__susceptible

    def run(self, susceptible_initial_percentage:float):
        if susceptible_initial_percentage > 1 or susceptible_initial_percentage < 0:
            raise ValueError("susceptible_initial_percentage has to be in range [0,1]")
        self.__vaccinated[0] = 0
        self.__recovered[0] = 0
        self.__susceptible[0] = susceptible_initial_percentage
        self.__infected[0] = 1 - self.__susceptible[0]
        for step in range(1, self.__steps):
            self.__susceptible[step] = self.__susceptible[step - 1] + self.__calc_curr_susceptible_change(step)
            self.__infected[step] = self.__infected[step - 1] + self.__calc_curr_infection_change(step)
            self.__recovered[step] = self.__recovered[step - 1] + self.__calc_curr_recovery_change(step)
            self.__vaccinated[step] = self.__vaccinated[step - 1] + self.__calc_curr_vaccinated_change(step)

    def __calc_curr_susceptible_change(self, step: int) -> int:
        return (self.__vaccination_decay_const * self.__vaccinated[step - 1]
                - self.__infection_const * self.__susceptible[step - 1] * self.__infected[step - 1]
                - self.__vaccination_const * self.__susceptible[step - 1]) * self.__dt

    def __calc_curr_infection_change(self, step: int) -> int:
        return (self.__infection_const * self.__susceptible[step - 1] * self.__infected[step - 1]
                + self.__vaccinated_infection_const * self.__vaccinated[step - 1] * self.__infected[step - 1]
                - self.__recovery_const * self.__infected[step - 1]) * self.__dt

    def __calc_curr_recovery_change(self, step: int) -> int:
        return (self.__recovery_const * self.__infected[step - 1]) * self.__dt

    def __calc_curr_vaccinated_change(self, step: int):
        return (self.__vaccination_const * self.__susceptible[step - 1] -
                self.__vaccination_decay_const * self.__vaccinated[step - 1] -
                self.__vaccinated_infection_const * self.__infected[step - 1] * self.__vaccinated[step - 1]) * \
               self.__dt


def plot_results(model: SIRModel):
    df = pandas.DataFrame(np.array([model.susceptible, model.infected, model.recovered, model.vaccinated]).T,
                          columns=['Susceptible', 'Infected', 'Recovered', 'Vaccinated'])
    plt = px.line(df,
                  title="Modeling Change in Different Population Groups During Pandemic",
                  labels={
                      "index": "Time",
                      "value": "Percent of Population",
                      "variable": "Group"
                  }
                  )
    plt.update_xaxes(showticklabels=False)
    plt.write_image(r"plots\SIR_plot.png")
    plt.show()


if __name__ == '__main__':
    sir = SIRModel(0.6, 0.15, 0.035, 0.015, 0.03, 1e-3)
    sir.run(0.999)
    plot_results(sir)
