import numpy as np


def make_ff(bead_type_list, bond_type_list, bead_radii_dict, filepath="ff.dat"):
    with open(filepath, 'w') as fout:
        for i, bead_type_1 in enumerate(bead_type_list):
            for bead_type_2 in bead_type_list[i:]:
                bead_type_1_sorted, bead_type_2_sorted = np.sort([bead_type_1, bead_type_2])
                bead_radii_1 = bead_radii_dict[bead_type_1_sorted]
                bead_radii_2 = bead_radii_dict[bead_type_2_sorted]
                kappa = 2.*(bead_radii_1**2 + bead_radii_2**2)
                make_gaussian(bead_type_1_sorted, bead_type_2_sorted, kappa, fout)
        for pair in bond_type_list:
            bead_type_1, bead_type_2 = np.sort(pair)
            make_bonded(bead_type_1, bead_type_2, fout)


def make_gaussian(bead_type_1, bead_type_2, kappa, fout):
    name = "Gaussian_{}{}".format(bead_type_1, bead_type_2)
    fout.write(">>> POTENTIAL {}\n".format(name))
    fout.write("{'Epsilon' : 0.0000e+00 ,\n")
    fout.write(" 'B' : __GAUSS_PREFACTOR__ ,\n")
    fout.write(" 'Kappa' : {:.5e} ,\n".format(kappa))
    fout.write(" 'Dist0' : 0.0000e+00 ,\n")
    fout.write(" 'Sigma' : 1.0000e+00 }\n")


def make_bonded(bead_type_1, bead_type_2, fout):
    name = "Bonded_{}{}".format(bead_type_1, bead_type_2)
    fout.write(">>> POTENTIAL {}\n".format(name))
    fout.write("{'Dist0' : __BOND_DIST0__ ,\n")
    fout.write(" 'FConst' : __BOND_FCONST__ }\n")


if __name__ == '__main__':
    bead_type_list = ['B4', 'C4', 'B12', 'C12', 'E12', 'D']
    bond_type_list = [('B4', 'B4'), ('B4', 'C4'), ('B4', 'B12'),
                      ('B12', 'B12'), ('B12', 'C12'), ('C12', 'E12'),
                      ('D', 'D')]
    bead_radii_dict = {}
    make_ff(bead_type_list, bond_type_list, bead_radii_dict)
