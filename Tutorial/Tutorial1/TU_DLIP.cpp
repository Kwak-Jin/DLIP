//
// Created by jinkwak on 2024-03-08.
//

#include "TU_DLIP.hpp"
#include <iostream>
int sum(int val1, int val2){
	return (val1+val2);
}

void Account::deposit(int money){
	balance += money;
}

void Account::withdraw(int money){
	balance -= money;
}

// Class Constructor 1
MyNum::MyNum(){}

// Class Constructor 2
MyNum::MyNum(int x){
	num = x;
}