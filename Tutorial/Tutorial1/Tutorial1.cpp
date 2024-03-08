/** @brief C++ Tutorial
*   @author Jin Kwak
*   @created 2024/03/08
*   @modified 2024/03/08
*/

#include <iostream>
#include "TU_DLIP.hpp"

template<typename T>
T Sum(T a, T b){
	return a+b;
}

int main(){
	int A = 10;
	int B = 20;
	int C = sum(A,B);
	std::cout<<"The Sum ="<<C<<std::endl;
	return 0;
}
