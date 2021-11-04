import numpy as np
import pandas
import plotly.express as px


TOT_DAYS = 150
INCUBATION_DAYS = 65
TREATMENT = True


class HIVModel:

    def __init__(self, infection_const: float, latent_to_infected: float,
                 uninfected_death_const: float, viruses_death_const, infected_death_const: float, latent_death_const,
                 uninfected_production_const, viruses_released_in_lysis_count):
        self.dt = 1e-2
        self.steps = int(TOT_DAYS / self.dt)
        self.latent_percent = 0.2
        self.latent_incubation = int(INCUBATION_DAYS / self.dt)
        self.is_latent_burst = False
        self.c_infection = infection_const
        self.c_latent_to_infected = latent_to_infected
        self.c_uninfected_death = uninfected_death_const
        self.c_infected_death = infected_death_const
        self.c_viruses_degrade = viruses_death_const
        self.c_latent_death = latent_death_const
        self.c_uninfected_production = uninfected_production_const
        self.c_viruses_released = viruses_released_in_lysis_count
        self.uninfected = np.empty(self.steps)
        self.latent = np.empty(self.steps)
        self.infected = np.empty(self.steps)
        self.viruses = np.empty(self.steps)

    def run(self, viruses_initial_population: float, healthy_t_cell_initial_population):
        self.uninfected[0] = healthy_t_cell_initial_population
        self.viruses[0] = viruses_initial_population
        self.latent[0] = 0
        self.infected[0] = 0
        for step in range(1, self.steps):
            if step == self.latent_incubation:
                self.is_latent_burst = not self.is_latent_burst
            self.uninfected[step] = self.uninfected[step - 1] + self.__calc_curr_uninfected_change(step)
            self.infected[step] = self.infected[step - 1] + self.__calc_curr_infected_change(step)
            self.viruses[step] = self.viruses[step - 1] + self.__calc_curr_viruses_change(step)
            self.latent[step] = self.latent[step - 1] + self.__calc_curr_latent_change(step)
        return

    def __calc_curr_uninfected_change(self, step: int) -> int:
        return (self.c_uninfected_production
                - self.c_infection * self.uninfected[step - 1] * self.viruses[step - 1]
                - self.c_uninfected_death * self.uninfected[step - 1]
                ) * self.dt

    def __calc_curr_infected_change(self, step: int) -> int:
        return (self.c_infection * self.uninfected[step - 1] * self.viruses[step - 1] * (1 - self.latent_percent)
                + self.c_latent_to_infected * self.latent[step - 1] * self.is_latent_burst
                - self.c_infected_death * self.infected[step - 1]
                ) * self.dt

    def __calc_curr_viruses_change(self, step: int) -> int:
        return (self.c_viruses_released * self.infected[step - 1] -
                self.c_viruses_degrade * self.viruses[step - 1]
                ) * self.dt

    def __calc_curr_latent_change(self, step: int):
        return (self.latent_percent * self.c_infection * self.uninfected[step - 1] * self.viruses[step - 1]
                - (self.c_latent_to_infected + self.c_latent_death) * self.latent[step - 1]
                * self.is_latent_burst
                ) * self.dt


def plot_results(model: HIVModel, days: int, fold: int):
    viruses_norm = 10
    df = pandas.DataFrame(np.array([model.uninfected, model.infected, model.viruses / viruses_norm, model.latent]).T,
                          columns=['Uninfected', 'Infected', f'Viruses / {viruses_norm}', 'Latent'])
    plt = px.line(df,
                  title=f"Modeling Change in Cell and Virus Groups in HIV Infection, "
                        f"Virus Degradation Fold Increase: {fold}",
                  labels={
                      "index": "Time (Days)",
                      "value": "Population Size",
                      "variable": "Group"
                  }
                  )
    plt.update_xaxes(tickvals=tuple(range(0, len(model.infected), int(5/model.dt))),
                     tickmode="array", ticktext=tuple(range(0, TOT_DAYS, 5)))
    plt.show()
    #print("writing image")
    #plt.write_image(rf"plots\HIV_plot_{days}.png")


if __name__ == '__main__':
    virus_fold = 9 if TREATMENT else 1
    hiv = HIVModel(infection_const=0.00027, latent_to_infected=0.05 * (1 + 9 * TREATMENT),
                   uninfected_death_const=0.00136, infected_death_const=0.33,
                   viruses_death_const=2 * (1 + (virus_fold - 1) * TREATMENT), latent_death_const=0.00136,
                   uninfected_production_const=0.272, viruses_released_in_lysis_count=100)
    hiv.run(viruses_initial_population=10, healthy_t_cell_initial_population=500)
    plot_results(hiv, INCUBATION_DAYS, virus_fold)
