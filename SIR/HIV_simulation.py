import numpy as np
import pandas
import plotly.express as px


DAYS = 20


class HIVModel:

    def __init__(self, infection_const: float, lysis_const: float, lysogenic_to_lytic_const: float,
                 uninfected_death_const: float, viruses_death_const, infected_death_const: float, latent_death_const,
                 uninfected_production_const, viruses_released_in_lysis_count, dt: float):
        self.__infection_const = infection_const
        self.__lysis_const = lysis_const
        self.__lysogenic_to_lytic_rate = lysogenic_to_lytic_const
        self.__uninfected_death_const = uninfected_death_const
        self.__infected_death_const = infected_death_const
        self.__viruses_death_const = viruses_death_const
        self.__latent_death_const = latent_death_const
        self.__uninfected_production_const = uninfected_production_const
        self.__steps = int(250 / dt)
        self.__uninfected = np.empty(self.__steps)
        self.__latent = np.empty(self.__steps)
        self.__infected = np.empty(self.__steps)
        self.__viruses = np.empty(self.__steps)
        self.__virueses_released_in_lysis = viruses_released_in_lysis_count
        self.__dt = dt
        self.__lysogenic_part_of_infections = 0.2
        self.__is_lysogenic_burst = False
        self.__lysogenic_incubation = int(DAYS * 2 / dt)

    @property
    def infected(self):
        return self.__infected

    @property
    def latent(self):
        return self.__latent

    @property
    def viruses(self):
        return self.__viruses

    @property
    def uninfected(self):
        return self.__uninfected

    def run(self, viruses_initial_population: float, healthy_t_cell_initial_population):
        self.__latent[0] = 0
        self.__viruses[0] = viruses_initial_population
        self.__uninfected[0] = healthy_t_cell_initial_population
        self.__infected[0] = 0
        for step in range(1, self.__steps):
            if step == self.__lysogenic_incubation:
                self.__is_lysogenic_burst = not self.__is_lysogenic_burst
            self.__uninfected[step] = self.__uninfected[step - 1] + self.__calc_curr_uninfected_change(step)
            self.__infected[step] = self.__infected[step - 1] + self.__calc_curr_infected_change(step)
            self.__viruses[step] = self.__viruses[step - 1] + self.__calc_curr_viruses_change(step)
            self.__latent[step] = self.__latent[step - 1] + self.__calc_curr_latent_change(step)

    def __calc_curr_uninfected_change(self, step: int) -> int:
        return (self.__uninfected_production_const
                - self.__infection_const * self.__uninfected[step - 1] * self.__viruses[step - 1]
                - self.__uninfected_death_const * self.__uninfected[step - 1]) * self.__dt

    def __calc_curr_infected_change(self, step: int) -> int:
        return (self.__infection_const * self.__uninfected[step - 1] * self.__viruses[step - 1]
                * (1 - self.__lysogenic_part_of_infections)
                + self.__lysogenic_to_lytic_rate * self.__latent[step - 1] * self.__is_lysogenic_burst
                - (self.__infected_death_const + self.__lysis_const) * self.__infected[step - 1]) * self.__dt

    def __calc_curr_viruses_change(self, step: int) -> int:
        return (self.__virueses_released_in_lysis * self.__lysis_const * self.__infected[step - 1] -
                self.__viruses_death_const * self.__viruses[step - 1]) * self.__dt

    def __calc_curr_latent_change(self, step: int):
        return (self.__lysogenic_part_of_infections * self.__infection_const * self.__uninfected[step - 1]
                * self.__viruses[step - 1]
                - (self.__lysogenic_to_lytic_rate + self.__latent_death_const) * self.__latent[step - 1]
                * self.__is_lysogenic_burst) * \
               self.__dt


def plot_results(model: HIVModel, days: int):
    df = pandas.DataFrame(np.array([model.uninfected, model.infected, model.viruses, model.latent]).T,
                          columns=['Uninfected', 'Infected', 'Viruses', 'Latent'])
    plt = px.line(df,
                  title=f"Modeling Change in Cell and Virus Groups in HIV Infection, Lysogenic Latency Time: {days} Days",
                  labels={
                      "index": "Time (Days)",
                      "value": "Population Size",
                      "variable": "Group"
                  }
                  )
    plt.update_xaxes(tickvals=tuple(range(0, len(model.infected), len(model.infected)//25)),
                     tickmode="array", ticktext=tuple(range(0, 126, 5)))
    plt.write_image(rf"plots\HIV_plot_{days}.png")
    plt.show()


if __name__ == '__main__':
    hiv = HIVModel(infection_const=0.000135, lysis_const=0.165, lysogenic_to_lytic_const=0.025,
                   uninfected_death_const=0.00068, infected_death_const=0.165, viruses_death_const=1,
                   latent_death_const=0.00068, uninfected_production_const=0.136,
                   viruses_released_in_lysis_count=50, dt=1e-3)
    hiv.run(viruses_initial_population=10, healthy_t_cell_initial_population=500)
    plot_results(hiv, DAYS)
