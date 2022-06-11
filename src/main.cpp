

#include <main.h>

using Eigen::MatrixXd;

int main(int argc, char** argv){
  MatrixXd mx(2,2);
  mx(0,0) = 10;
  std::cout << mx << std::endl;
  return 0;
}