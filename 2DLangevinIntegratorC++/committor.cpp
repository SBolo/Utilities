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
    int ntrajs = 1000;

    /** SIMULATION PARAMETERS */
    int total_time = 4000; //total number of timesteps to simulate
    int stride = 1; //frames will be saved once every 'stride'
    double timestep = 0.02; //integration timestep
    int len = (int)(total_time/stride);
    int ntransitions = 0;

    /** PHYSICAL PARAMETERS */
    double friction = 1.; //friction constant
    double kbT      = 0.2; //energy in k Boltzmann times the temperature

    vector<vector<double> > committor(100, vector<double>(100));

    double xmin = -0.1;
    double xmax = 0.1;
    double ymin = 1.3;
    double ymax = 1.6;

    vector<double> target(dim);
    vector<double> x0(dim);

    target[0] = 1.;
    target[1] = 0.;
    RandomNumbers *r = new RandomNumbers();

    for(int i=0; i<ntrajs; i++) {
        x0[0] = (xmax-xmin)*r->uniform() + xmin;
        x0[1] = (ymax-ymin)*r->uniform() + ymin;

        //allocate the class for the integrator
        Ito_integrator ito = Ito_integrator(r, dim, timestep, friction, kbT, x0);
        //loop over timesteps
        for(int j = 0; j < total_time; j++) {
            ito.evolve_committor(target, -2.5, 0.02);
            bool flag = ito.get_flag();
            if(flag) {
                ntransitions++;
                break;
            }
        }
    }

    cout << "Committor: " << (1.*ntransitions)/ntrajs << endl;
    return 0;
}
