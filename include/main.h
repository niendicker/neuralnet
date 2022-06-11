
/**
 * @file   main.h
 * @author Marcelo Niendicker Grando
 * @date   2022/06/09
 * @brief  Basic neural network algorithms
 */ 
#ifndef NN_MAIN_H
#define NN_MAIN_H

#include <iostream>
#include <string>

#include <neuralnet.h>

using namespace std;

class base
{
private:
  string* a;
public:
  base(/* args */);
  ~base();
};

base::base(/* args */)
{
  cout << "cazzo";
}

base::~base()
{
}

#endif/* main */

