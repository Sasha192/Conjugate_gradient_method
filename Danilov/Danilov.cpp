#include <iostream>
#include <cmath>
#include "pch.h"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/io.hpp"

double eps = 1e-3;
using namespace boost::numeric::ublas;



typedef  double(*Function)(vector<double>);


double F1(vector<double> x)
{
	double x1 = x[0];
	double x2 = x[1];
	double x3 = x[2];
	return std::exp(pow(x1,2) + pow(x2,2)) + std::log(4 + pow(x2,2) + 2*pow(x3,2));
}

vector<double> GradF1(vector<double> x)
{
	vector<double> res(x.size());
	double x1 = x[0];
	double x2 = x[1];
	double x3 = x[2];
	res[0] = 2 * x1*(std::exp(pow(x1, 2) + pow(x2, 2)));
	res[1] = 2 * x2*(std::exp(pow(x1, 2) + pow(x2, 2))) + (2 * x2) / (pow(x2, 2) + 2 * pow(x3, 2) + 4);
	res[2] = 4 * x3 / (pow(x2, 2) + 2 * pow(x3, 2) + 4);
	return res;
}

double FQuad(vector<double> x)
{
	return (x[0] - 1) * (x[0] - 1) + (x[1] - 2) * (x[1] - 2);
}

vector<double> GradFQuad(vector<double> x)
{
	vector<double> res(x.size());
	res[0] = 2 * (x[0] - 1);
	res[1] = 2 * (x[1] - 2);
	return res;
}

double FindMin(Function F, vector <double> s, vector<double> p)
{
	const double EPS = 1e-8;
	double a = 0;
	double b = 1e5;
	double x0 = a + 0.5 * (3 - sqrt(5.0)) * (b - a);
	double x1 = b - x0 + a;

	while (abs(b - a) > EPS)
	{
		if (F(s + x0 * p) < F(s + x1 * p))	b = x1;
		else a = x0;

		x1 = x0;
		x0 = b + a - x1;
	}
	return (a + b) / 2;
}

void PrintSolution(vector<double> x, double val, int numIter)
{
	std::cout << "-----------------------" << std::endl;
	std::cout << "Number of iterations: " << numIter << std::endl;
	std::cout << "Computed solution: " << std::endl;
	std::cout << x << std::endl;
	std::cout << "Function value: " << val << std::endl << std::endl;
}

vector<double> FletcherRievesMethod(int spaceSize, Function F, vector<double>(*GradF) (vector<double>))
{
	//F - указатель на минимизируемую  функцию
	//GradF - указатель на градиент F
	const double EPS = 1e-5;
	//Начальное приближение
	vector<double> x = zero_vector<double>(spaceSize);
	double curVal = F(x);
	double prevVal = curVal;
	vector<double> p = -GradF(x);
	double gradSquare = inner_prod(p, p);

	int numIter = 0;
	do
	{
		numIter++;
		double alpha, beta, newGradSquare;
		vector<double> newGrad;


		//Ищем минимум F(x + alpha * p) с помощью метода одномерной оптимизации
		alpha = FindMin(F, x, p);
		x = x + alpha * p;

		newGrad = -GradF(x);
		newGradSquare = inner_prod(newGrad, newGrad);

		if (numIter % (5 * spaceSize) == 0) beta = 0; //Обновление
		else
			beta = newGradSquare / gradSquare; //Используем метод Флетчера - Ривса
		p = newGrad + beta * p;

		prevVal = curVal;
		curVal = F(x);

		gradSquare = newGradSquare;
	} while (gradSquare > eps);

	std::cout << "Fletcher-Rieves Method: " << std::endl;
	PrintSolution(x, F(x), numIter);
	return x;

}

vector<double> PolakRibiereMethod(int spaceSize, Function F, vector<double>(*GradF) (vector<double>))
{
	//F - указатель на минимизируемую  функцию
	//GradF - указатель на градиент F
	const double EPS = 1e-5;
	//Начальное приближение
	vector<double> x = zero_vector<double>(spaceSize);
	double curVal = F(x);
	double prevVal = curVal;
	vector<double> grad = -GradF(x);
	vector<double> p = -grad;
	double gradSquare = inner_prod(p, p);

	int numIter = 0;
	do
	{
		numIter++;
		double alpha, beta, newGradSquare;
		vector<double> newGrad;

		//Ищем минимум F(x + alpha * p) с помощью метода одномерной оптимизации
		alpha = FindMin(F, x, p);
		x = x + alpha * p;

		newGrad = -GradF(x);
		newGradSquare = inner_prod(newGrad, newGrad);

		if (numIter % (5 * spaceSize) == 0) beta = 0; //Обновление
		else
			beta = -(newGradSquare - inner_prod(newGrad, grad)) / gradSquare; //Используем метод Флетчера - Ривса
		p = newGrad + beta * p;

		prevVal = curVal;
		curVal = F(x);

		grad = newGrad;
		gradSquare = newGradSquare;
	} while (gradSquare > eps);

	std::cout << "Polak-Ribiere Method: " << std::endl;
	PrintSolution(x, F(x), numIter);
	return x;

}

int main()
{
	std::cout.precision(9);
	FletcherRievesMethod(3, &F1, &GradF1);
	PolakRibiereMethod(3, &F1, &GradF1);

	return 0;
}

