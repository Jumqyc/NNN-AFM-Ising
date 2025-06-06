#ifndef _ISING_H
#define _ISING_H
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
using namespace std;
using std::invalid_argument;

class Ising
{
public:
    // constructor
    Ising(int a, int b) : a(a), b(b)
    {
        spin.resize(a, vector<int>(b, 1)); // initialize all spins to +1
        if (a % 2 != 0 || b % 2 != 0)
        {
            throw invalid_argument("Lattice size must be even.");
        }
        e_data = vector<float>();
        m_data = vector<int>();
        afm_data = vector<int>();
        num_samples = 0;

        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < b; j++)
            {
                spin[i][j] = (i + j) % 2 == 0 ? 1 : -1; // afm ground state initialization
            }
        }
    }

    void set_parameters(float temperature, float J1, float J2, float J3)
    {
        t = temperature;
        J_1 = J1;
        J_2 = J2;
        J_3 = J3;
        expJ1 = exp(-2 * abs(J_1) / t);
        expJ2 = exp(-2 * abs(J_2) / t);
        expJ3 = exp(-2 * abs(J_3) / t);
        update_thermal(); // update the thermal properties
    }

    void set_spin(const vector<vector<int>> &new_spin)
    {
        spin = new_spin;
    }

    void set_collected_data(const vector<float> &ext_e_data,
                            const vector<int> &ext_m_data,
                            const vector<int> &ext_afm_data)
    {
        e_data = ext_e_data;
        m_data = ext_m_data;
        afm_data = ext_afm_data;
        num_samples = e_data.size();
    }
    // core functions

    void run_local(int Ntest, int spacing)
    {
        num_samples += Ntest;
        e_data.reserve(num_samples);
        m_data.reserve(num_samples);
        afm_data.reserve(num_samples);
        for (int i = 0; i < Ntest; i++)
        {
            local_update(spacing); // perform local update with specified spacing
            e_data.push_back(e);
            m_data.push_back(m);
            afm_data.push_back(afm);
        }
    }

    void run_cluster(int Ntest, int spacing)
    {
        num_samples += Ntest;
        e_data.reserve(num_samples);
        m_data.reserve(num_samples);
        afm_data.reserve(num_samples);
        for (int i = 0; i < Ntest; i++)
        {
            cluster_update(spacing); // perform cluster update with specified spacing
            update_thermal();        // update the thermal properties after the cluster update
            e_data.push_back(e);
            m_data.push_back(m);
            afm_data.push_back(afm);
        }
    }

    void random_seed()
    {
        // reseed the random number generator
        static thread_local std::mt19937 rng(std::random_device{}());
        rng.seed(std::random_device{}());
    }

    vector<float> get_energy() const { return e_data; }
    vector<int> get_magnetization() const { return m_data; }
    vector<int> get_afm() const { return afm_data; }
    vector<vector<int>> get_spin() const { return spin; }

    float temperature() const { return t; }
    float J1() const { return J_1; }
    float J2() const { return J_2; }
    float J3() const { return J_3; }
    int size_x() const { return a; }
    int size_y() const { return b; }

private:
    const int a, b;            // dimensions of the lattice
    vector<vector<int>> spin;  // spin configuration
    float t;                   // temperature
    float J_1, J_2, J_3;       // couplings
    float expJ1, expJ2, expJ3; // precomputed exponential for efficiency
    float e;
    int m, afm; // thermaldynamic properties
    vector<float> e_data;
    vector<int> m_data, afm_data; // data collection

    int num_samples; // number of samples collected

    // functions
    void update_thermal()
    {
        e = 0;
        m = 0;
        afm = 0;

        // initialize the physical quantities
        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < b; j++)
            {
                e += -spin[i][j] * (J_1 * (spin[(i + 1) % a][j] + spin[(i - 1 + a) % a][j]) +
                                    J_2 * (spin[i][(j + 1) % b] + spin[i][(j - 1 + b) % b]) +
                                    J_3 * (spin[(i + 1) % a][(j + 1) % b] + spin[(i - 1 + a) % a][(j - 1 + b) % b] +
                                           spin[(i - 1 + a) % a][(j + 1) % b] + spin[(i + 1) % a][(j - 1 + b) % b]));
                m += spin[i][j];
                afm += (2 * ((i + j) % 2) - 1) * spin[i][j];
            }
        }
        e /= 2.0; // divide by 2 because we counted each bond twice
    }

    // generates random integers between [0, upper)
    int rand_int(int upper)
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, upper - 1);
        return dist(rng);
    }

    // generate a random number between 0 and 1
    float uniform()
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        return dist(rng);
    }

    void local_update(int spacing)
    {
        for (int i = 0; i < spacing * a * b; i++)
        {
            // pick a random spin
            int x = rand_int(a);
            int y = rand_int(b);

            int xup = (x + 1) % a;
            int xdown = (x - 1 + a) % a;
            int yup = (y + 1) % b;
            int ydown = (y - 1 + b) % b;

            // calculate the energy change
            float dE = 2 * spin[x][y] * (J_1 * (spin[xup][y] + spin[xdown][y]) + J_2 * (spin[x][yup] + spin[x][ydown]) + J_3 * (spin[xup][yup] + spin[xdown][ydown] + spin[xdown][yup] + spin[xup][ydown]));

            // Metropolis algorithm
            if (uniform() < exp(-dE / t))
            {
                int s = spin[x][y] *= -1; // flip the spin
                e += dE;
                m += 2 * s;
                afm += (4 * ((x + y) % 2) - 2) * s;
            }
        }
    }

    void cluster_update(int spacing)
    {
        vector<vector<bool>> in_cluster(a, vector<bool>(b, false));

        vector<pair<int, int>> stack;
        stack.reserve(a * b); // reserve space for the stack to avoid reallocations

        for (int i = 0; i < spacing; i++)
        {

            // Efficiently reset the in_cluster matrix

            // seed of the cluster
            int x0 = rand_int(a);
            int y0 = rand_int(b);

            in_cluster[x0][y0] = true; // start with a random spin
            stack.push_back(make_pair(x0, y0));

            while (!stack.empty())
            {
                auto [x, y] = stack.back();
                stack.pop_back();
                int xup = (x + 1) % a;
                int xdown = (x - 1 + a) % a;
                int yup = (y + 1) % b;
                int ydown = (y - 1 + b) % b;
                int s = spin[x][y];

                // check the neighbors and add them to the cluster if they are not already in it
                if (!in_cluster[xup][y] && spin[xup][y] != s) //(x+1,y)
                {
                    if (expJ1 < uniform())
                    {
                        in_cluster[xup][y] = true;
                        stack.push_back(make_pair(xup, y));
                    }
                }
                if (!in_cluster[xdown][y] && spin[xdown][y] != s) //(x-1,y)
                {
                    if (expJ1 < uniform())
                    {
                        in_cluster[xdown][y] = true;
                        stack.push_back(make_pair(xdown, y));
                    }
                }
                if (!in_cluster[x][yup] && spin[x][yup] != s) //(x,y+1)
                {
                    if (expJ2 < uniform())
                    {
                        in_cluster[x][yup] = true;
                        stack.push_back(make_pair(x, yup));
                    }
                }
                if (!in_cluster[x][ydown] && spin[x][ydown] != s) //(x,y-1)
                {
                    if (expJ2 < uniform())
                    {
                        in_cluster[x][ydown] = true;
                        stack.push_back(make_pair(x, ydown));
                    }
                }
                if (!in_cluster[xup][yup] && spin[xup][yup] == s) //(x+1,y+1)
                {
                    if (expJ3 < uniform())
                    {
                        in_cluster[xup][yup] = true;
                        stack.push_back(make_pair(xup, yup));
                    }
                }
                if (!in_cluster[xdown][yup] && spin[xdown][yup] == s) //(x-1,y+1)
                {
                    if (expJ3 < uniform())
                    {
                        in_cluster[xdown][yup] = true;
                        stack.push_back(make_pair(xdown, yup));
                    }
                }
                if (!in_cluster[xdown][ydown] && spin[xdown][ydown] == s) //(x-1,y-1)
                {
                    if (expJ3 < uniform())
                    {
                        in_cluster[xdown][ydown] = true;
                        stack.push_back(make_pair(xdown, ydown));
                    }
                }
                if (!in_cluster[xup][ydown] && spin[xup][ydown] == s) //(x+1,y-1)
                {
                    if (expJ3 < uniform())
                    {
                        in_cluster[xup][ydown] = true;
                        stack.push_back(make_pair(xup, ydown));
                    }
                }
            }
        }
        // now the stack is empty and the cluster is formed
        // so now we can repeat all the process
        for (auto &row : in_cluster)
        {
            std::fill(row.begin(), row.end(), false);
        }
    }
};
#endif