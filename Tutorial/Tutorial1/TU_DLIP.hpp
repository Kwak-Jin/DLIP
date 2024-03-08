//
// Created by jinkwak on 2024-03-08.
//

#ifndef DLIP_TU_DLIP_HPP
#define DLIP_TU_DLIP_HPP
#include <iostream>

int sum(int val1, int val2);

/*  Class Definition  */
class Account{
public:
	char number[20];
	char password[20];
	char name[20];
	int balance;
	void deposit(int money);  // Can include functions
	void withdraw(int money);  // Can include function
};

class MyNum{
public:
	MyNum();  // Constructor 1
	MyNum(int x);  // Constructor 2
	int num;
};

#endif //DLIP_TU_DLIP_HPP
