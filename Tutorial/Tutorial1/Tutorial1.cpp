/**
*
*/

#include <iostream>

double sum(double a, double b);

int main(){
	double A = 10.;
	double B = 20.;
	double C = sum(A,B);
	std::cout<<"The Sum ="<<C<<std::endl;
	return 0;
}

double sum(double a,double b){
	return a+b;
}