#include "Ito_integrator.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

/** CONSTRUCTOR FOR rMD **/
Ito_integrator::Ito_integrator(RandomNumbers *r, int dmn, double tmstp, double gamma, double kT, vector<double> x0) {

    rng = r;
    dt = tmstp;
    adim_dt = (dt/gamma);
    variance = sqrt(2. * kT * adim_dt);
    dim = dmn;
    time = 0.;

    position = vector<double>(dim);
    f = vector<double>(dim);

    for(int d = 0; d < dim; d++) {
        position[d] = x0[d];
    }

    stop_flag = false;
}
//------------------------------------------------------------------------------

Ito_integrator::~Ito_integrator() {
}
//------------------------------------------------------------------------------

double Ito_integrator::d_hypot(vector<double> vec) {

    double tmp = 0;
    for( int d = 0; d < dim; d++ ) {
        tmp += vec[d] * vec[d];
    }

    return sqrt(tmp);
}
//------------------------------------------------------------------------------

double Ito_integrator::calc_z(vector<double> target) {

    double dist = 0, pos, trg;

        for( int d = 0; d < dim; d++ ) {
            pos = position[d];
            trg = target[d];
            dist += (pos - trg) * (pos - trg);
        }

    return sqrt(dist);
}
//------------------------------------------------------------------------------

void Ito_integrator::force() {

    /*
     * -gradV(x) where V(x) is the three-weels potential
     * Analytical form and parameters were taken from
     * Philipp Metznera, Christof Schütteb, Eric Vanden-Eijnden "Illustration of transition path
     * theory on a collection of simple examples"
     */

    double x = position[0];
    double y = position[1];

    double x2 = x*x;
    double xm1 = (x-1)*(x-1);
    double xp1 = (x+1)*(x+1);
    double y2 = y*y;
    double yc = (y-5./3.)*(y-5./3.);
    double yu = (y-1./3.)*(y-1./3.);

    f[0] = -10*(x-1)*exp(-xm1 -y2) + x*exp(-x2)*( -6*exp(-yc) + 10*exp(-y2)) - 10*(x+1)*exp(-xp1-y2) - 0.8*x*x2;
    f[1] = -6*(y-5./3.)*exp(-x2-yc) + 10*y*exp(-x2-y2) - 10*y*exp(-y2)*(exp(-xm1)+exp(-xp1)) - 0.8*yu*(y-1./3.);

}
//------------------------------------------------------------------------------

void Ito_integrator::force_potential() {

    double x = position[0];
    double y = position[1];

    double x2 = x*x;
    double xm1 = (x-1)*(x-1);
    double xp1 = (x+1)*(x+1);
    double y2 = y*y;
    double yc = (y-5./3.)*(y-5./3.);
    double yu = (y-1./3.)*(y-1./3.);
    double x2yc = exp(-x2-yc);
    double xm1y2 = exp(-xm1-y2);
    double xp1y2 = exp(-xp1-y2);
    double x2y2 = exp(-x2-y2);
    double ey2 = exp(-y2);

    f[0] = -10*(x-1)*xm1y2 + x*exp(-x2)*(-6*exp(-yc) + 10*ey2) - 10*(x+1)*xp1y2 - 0.8*x*x2;
    f[1] = -6*(y-5./3.)*x2yc + 10*y*x2y2 - 10*y*ey2*(exp(-xm1)+exp(-xp1)) - 0.8*yu*(y-1./3.);
    p = 5. * (x2y2 - 3./5. * x2yc - xm1y2 - xp1y2) + 1./5. * (x2*x2 + yu*yu);
}

void Ito_integrator::evolve() {

    force();
    for( int d = 0; d < dim; d++ ) {
        position[d] += f[d] * adim_dt + variance * rng->gaussian(1.);
    }

    force_mod = d_hypot(f);
    force_mod *= force_mod;
    time += dt;
}
//------------------------------------------------------------------------------

void Ito_integrator::evolve_committor(vector<double> target, double pot_thresh, double z_thresh) {

    force_potential();
    for( int d = 0; d < dim; d++ ) {
        position[d] += f[d] * adim_dt + variance * rng->gaussian(1.);
    }
    double z = calc_z(target);

    if( p < pot_thresh and z < z_thresh ) {
        stop_flag = true;
    }

}
//------------------------------------------------------------------------------

bool Ito_integrator::get_flag() {
    return stop_flag;
}

double Ito_integrator::get_position(int i) {

    if( i > dim ) {
        return 0;
    } else {
        return position[i];
    }
}
//------------------------------------------------------------------------------

double Ito_integrator::get_force() {
    return force_mod;
}
//------------------------------------------------------------------------------
