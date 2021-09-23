#include "Ito_integrator.h"
#include <fstream>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

int main(int argc, char *argv[]) {
    ofstream info;
    char filename[50];
    double bfm, temp, x, y, z1, z2, z3;
    int dim = 2;
    vector<double> x0(dim); //vector for initial condition
    x0[0] = -1.;
    x0[1] = 0.;

    /** SIMULATION PARAMETERS */
    int total_time = 100000000; //total number of timesteps to simulate
    int stride = 100; //frames will be saved once every 'stride'
    double timestep = 0.02; //integration timestep
    int len = (int)(total_time/stride);

    /** PHYSICAL PARAMETERS */
    double friction = 1.; //friction constant
    double kbT      = 1/6.67; //energy in k Boltzmann times the temperature

    /** TARGETS FOR Z CALCULATION */
    vector<double> t1(dim);
    vector<double> t2(dim);
    vector<double> t3(dim);
    t1[0] = -1.;
    t1[1] = 0.;
    t2[0] = 1.5;
    t2[1] = 0.;
    t3[0] = 1.;
    t3[1] = 0.;

    cout << endl;
    cout << "# LANGEVIN DYNAMICS IN A 3 WELL POTENTIAL" << endl;
    cout << "# ---------------------------------------" << endl;
    cout << "# X0:     " << "(" << x0[0] << ", " << x0[1] << ")" << endl;
    cout << "# STEPS:  " << total_time << endl;
    cout << "# STRIDE: " << stride << endl;
    cout << "# DT:     " << timestep << endl;
    cout << "# GAMMA:  " << friction << endl;
    cout << "# KT:     " << kbT << endl;

    /** RNG INITIALIZATION */
    RandomNumbers *rng = new RandomNumbers();

    //allocate the class for the integrator
    Ito_integrator ito = Ito_integrator(rng, dim, timestep, friction, kbT, x0);

    z1 = ito.calc_z(t1);
    z2 = ito.calc_z(t2);
    z3 = ito.calc_z(t3);

    //save trajectory
    sprintf(filename, "traj.txt");
    info.open(filename, ios::out);
    info << "#x\t y\t z1\t z2 \tz3" << endl;
    info << x << "\t" << y << "\t" << z1 << "\t" << z2 << "\t" << z3 << endl;

    //loop over timesteps
    for(int j = 0; j < total_time; j++) {
        ito.evolve();

        //once every stride, collect information and print it on file
        if( j%stride == 0 ) {
            z1 = ito.calc_z(t1);
            z2 = ito.calc_z(t2);
            z3 = ito.calc_z(t3);
            x = ito.get_position(0);
            y = ito.get_position(1);
            info << x << "\t" << y << "\t" << z1 << "\t" << z2 << "\t" << z3 << endl;
        }
    }

    /** LOOP OVER TIMES ENDS HERE **/
     info.close(); //<-------- FOR INFOS
     cout << "# DONE!" << endl;
    return 0;
}
