import numpy as np
import random
import json
import matplotlib.pyplot as plt
from ase.units import kcal, mol, J

file = '/Users/xiaowei_xie/PycharmProjects/sugar/SEI/SEI_elementary_reactions_30kcal_no_gas_fast_ring_opening_fast_reduction_separate_unibi_mol_faster_uni_concerted_new.json'
with open(file) as data:
    reaction_file = json.load(data)

with open('/Users/xiaowei_xie/PycharmProjects/electrolyte/one_bond_reactions/one_bond_rxns.json') as data_file:
    one_rxns_dict = json.load(data_file)

with open('/Users/xiaowei_xie/PycharmProjects/electrolyte/all_data_in_database/all_species_energy_dict.json') as data_file:
    all_species_energy_dict = json.load(data_file)


num_species = len(all_species_energy_dict.keys())
#4720
nums = ['y('+ str(i + 1)+')' for i in range(num_species+1)]
nums_to_key = {}
key_to_nums = {}
for i in range(num_species):
    nums_to_key[nums[i]] = list(all_species_energy_dict.keys())[i]
    key_to_nums[list(all_species_energy_dict.keys())[i]] = nums[i]
nums_to_key['y('+str(num_species+1)+')'] = 'e_-1'
key_to_nums['e_-1'] = 'y('+str(num_species+1)+')'

all_species_energy_dict['e_-1'] = -0.079011


class Stochastic_Simulation_Dummy:

    def __init__(self):
        self.num_species = len(nums)
        self.num_reactions = len(reaction_file['reactions'])
        self.reactions = reaction_file['reactions']
        self.reaction_rates = []
        for i, rxn in enumerate(self.reactions):
            rate = float(self.reactions[i][0])
            self.reaction_rates.append(rate)
        return

    def get_propensities(self,num_of_mols):
        self.propensities = []
        for i, rxn in enumerate(self.reactions):
            propensity = self.reaction_rates[i]
            reactants = rxn[1].keys()
            for reactant in reactants:
                num_of_reactants = rxn[1][reactant]
                reac = reactant.split('(')[1].split(')')[0]
                if num_of_reactants == 1:
                    propensity *= num_of_mols[int(reac)-1]
                elif num_of_reactants == 2:
                    propensity *= 0.5 * num_of_mols[int(reac)-1] * (num_of_mols[int(reac)-1] - 1)
                elif num_of_reactants == 3:
                    propensity *= 1 / 6 * num_of_mols[int(reac)-1] * (num_of_mols[int(reac)-1] - 1) \
                                  * (num_of_mols[int(reac)-1] - 2)
            self.propensities.append(propensity)
        return self.propensities

    def direct_method(self,initial_conc, time_span, max_output_length):
        '''
        :param initial_conc
        :param time_span: int, number of time steps
        :param max_output_length
        Gillespie, D.T. (1977) Exact Stochastic Simulation of Coupled Chemical Reactions. J Phys Chem, 81:25, 2340-2361.
        :return: t: time vector (Nreaction_events x 1); x:  species amounts (Nreaction_events x Nspecies);
                 rxns: rxns that have been fired up (Nreaction_events x 1).
        '''
        records = {}
        records['a'] = []
        records['random_rxn'] = []
        records['a0'] = []
        records['mu'] = []
        t = np.zeros(max_output_length)
        x = np.zeros([max_output_length, self.num_species])
        rxns = np.zeros(max_output_length)
        t[0] = 0
        x[0, :] = initial_conc
        rxn_count = 0

        while t[rxn_count] < time_span:
            a = self.get_propensities(x[rxn_count, :])
            a0 = np.sum(a)
            random_t = random.uniform(0, 1)
            random_rxn = random.uniform(0, 1)
            tau = -np.log(random_t) / a0
            mu = np.where(np.cumsum(a) >= random_rxn * a0)[0][0]
            records['a'].append(a)
            records['random_rxn'].append(random_rxn)
            records['a0'].append(a0)
            records['mu'].append(mu)

            if rxn_count + 1 > max_output_length:
                t = t[:rxn_count]
                x = x[:rxn_count]
                print(
                    "WARNING:Number of reaction events exceeded the number pre-allocated. Simulation terminated prematurely.")

            t[rxn_count + 1] = t[rxn_count] + tau
            x[rxn_count + 1] = x[rxn_count]
            current_reaction = self.reactions[mu]
            reactants = current_reaction[1].keys()
            #reactants = [reac.split('(')[1].split(')')[0] for reac in reactants]
            products = [key for key in current_reaction[2].keys() if current_reaction[2][key] > 0]
            #products = [prod.split('(')[1].split(')')[0] for prod in products]
            for reac in reactants:
                x[rxn_count + 1, int(reac.split('(')[1].split(')')[0])-1] -= current_reaction[1][reac]
            for prod in products:
                x[rxn_count + 1, int(prod.split('(')[1].split(')')[0])-1] += current_reaction[2][prod]

            rxns[rxn_count + 1] = mu
            rxn_count += 1

        t = t[:rxn_count]
        x = x[:rxn_count, :]
        rxns = rxns[:rxn_count]
        if t[-1] > time_span:
            t[-1] = time_span
            x[-1, :] = x[rxn_count - 1, :]
            rxns[-1] = rxns[rxn_count - 1]
        return t, x, rxns, records

if __name__ == '__main__':
    SS = Stochastic_Simulation_Dummy()
    initial_conc = np.zeros(SS.num_species)
    initial_conc[120] = 15000
    initial_conc[277] = 1000
    initial_conc[21] = 30
    initial_conc[4720] = 1000

    t, x, rxns,records = SS.direct_method(initial_conc, 1000000, 10000000)

    sorted_species_index = np.argsort(x[-1, :])[::-1]
    fig, ax = plt.subplots()
    for i in range(100):
        species_index = sorted_species_index[i]
        if x[-1, int(species_index)] > 0 and int(species_index) != 120 and int(species_index) != 277 and int(
                species_index) != 4720:
            ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
            # ax.plot(T,X[:,int(species_index)], 'C0o', alpha=0.5)
    plt.title('test')
    plt.legend(loc='upper left')
    plt.show()

    rxns_set = list(set(rxns))
    rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
    index = np.argsort(rxns_count)[::-1]
    sorted_rxns = np.array(rxns_set)[index]
    x0 = np.arange(len(rxns_set))
    fig, ax = plt.subplots()
    plt.bar(x0, rxns_count)
    #plt.xticks(x, ([str(int(rxn)) for rxn in rxns_set]))
    plt.show()

    for rxn in sorted_rxns:
        rxn = int(rxn)
        print(SS.reactions[rxn], SS.reaction_rates[rxn])
    non_zero_index = [i for i in range(len(records['a'][0])) if records['a'][0][i] != 0]
    non_zero_as = [records['a'][0][i] for i in range(len(records['a'][0])) if records['a'][0][i]!=0]
