#include<iostream>
#include<string>
#include<list>
#include<algorithm>
#include<fstream>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include <random>
#include<ctime>
using namespace std;

vector<vector<double> > MatrixMult(vector<vector<double>> a,vector<vector<double>> b)//return res=a*b,矩阵乘法
{
	int n1=a.size();
	int n2=a[0].size();
	int n3=b.size();
	int n4=b[0].size();

	vector<vector<double>> res(n1,vector<double>(n4,0));

	if(n2!=n3)
	{
		cout<<"错误:规模不匹配"<<endl;
		system( "PAUSE ");
		return res;
	}
	else
	{
		for(int i=0; i<n1; i++)
		{
			for(int j=0; j<n4; j++)
			{
				double temp=0;
				for(int k=0; k<n2; k++)
				{
					temp+=a[i][k]*b[k][j];
				}
				res[i][j]=temp;
			}
		}
	}
	return res;
}

vector<vector<double> > transform(vector<vector<double>> a)		//转置
{
	int n1=a.size();
	int n2=a[0].size();
	vector<vector<double>> temp(n2,vector<double>(n1,0));
	for(int i=0; i<n2; i++)
	{
		for(int j=0; j<n1; j++)
		{
			temp[i][j]=a[j][i];
		}
	}

	return temp;
}


double costfun(vector<vector<double>> x,vector<vector<double>> y,vector<vector<double>> w)	//代价函数
{
	int num=y.size();		//数据的个数
	int size=w.size();		//w的维数 ,系数个数
	double res=0;
	vector<vector<double>> temp;
	temp=MatrixMult(x,transform(w));
	for(int i=0; i<temp.size(); i++)
	{
		for(int j=0; j<temp[i].size(); j++)
		{
			res+=(temp[i][j]-y[i][j])*(temp[i][j]-y[i][j]);
		}
	}

	return res/num;
}

vector<double> SGD(vector<vector<double>> x,vector<vector<double>> y,vector<vector<double>> &w,double alpha,int epoch)
{
	int num=y.size();		//数据的个数
	int size=w[0].size();		//w的维数,系数个数
	vector<double> cost;


	//		w=w-(alpha/num)*(x*w.T-y).T*x;
	for(int i=0; i<epoch; i++)
	{

		
		vector<vector<double>> tempx;
		vector<vector<double>> tempy;

		for(int i=0; i<x.size(); i++)
		{
			double yes=0;
			yes = (rand()%10)/10.0;
			if(yes<=0.6)				//60%的概论选择
			{
				tempx.push_back(x[i]);
				tempy.push_back(y[i]);
			}
		}
		if(tempx.empty())
		{
			tempx.push_back(x[0]);
			tempy.push_back(y[0]);
		}
//		for(int row=0; row<tempx.size(); row++)
//		{
//			for(int col=0; col<tempx[row].size(); col++)
//			{
//				cout<<tempx[row][col]<<" ";
//			}
//			cout<<endl;
//		}

		vector<vector<double>> temp(w);

		temp=MatrixMult(tempx,transform(w));

		for(int row=0; row<temp.size(); row++)
		{
			for(int col=0; col<temp[row].size(); col++)
			{
				temp[row][col]-=tempy[row][col];
			}
		}

		temp=MatrixMult(transform(temp),tempx);

		for(int row=0; row<temp.size(); row++)
		{
			for(int col=0; col<temp[row].size(); col++)
			{
				w[row][col]-=(alpha/double(num))*temp[row][col];
			}
		}
		cost.push_back(costfun(x,y,w));
		if(i%(epoch/100)==0||i==epoch-1)					//每100次输出一次 
		{
			cout<<"epoch:"<<i+1<<endl;
			for(int row=0; row<temp.size(); row++)
			{
				for(int col=0; col<temp[row].size(); col++)
				{

					cout<<"w[row][col]:"<<w[row][col]<<" ";
				}
				cout<<endl;
			}
			cout<<"cost[i]="<<cost[i]<<endl<<endl;
		}
		if(cost[i]<0.0000001)			//提前结束
			break;
	}
	return cost;
}
vector<vector<double>> LinearRegression(vector<vector<double>> x,vector<vector<double>> y,double alpha,int epoch)
{
	for(int i=0; i<x.size(); i++)			//加上常数项
	{
		x[i].push_back(1);
	}
	vector<vector<double>> w(1,vector<double>(x[0].size()));		//随机生成系数，范围0-2
	srand((unsigned int)(time(NULL)));
	for(int i=0; i<w[0].size(); i++)
	{
		w[0][i]=double(rand() % 20 + 0)/10;
	}


	SGD(x,y,w,alpha,epoch);		//梯度下降


	vector<double> cost;

	cout<<endl<<"Y=";			//输出公式
	for(int i=0; i<w[0].size()-1; i++)
	{
		cout<<w[0][i]<<"X"<<i+1<<"+";
	}
	cout<<w[0][w[0].size()-1]<<endl;
	return w;
}
int main()
{
	double alpha = 0.0001;
	int epoch = 1000000;
	vector<vector<double>> x= {{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13}};
	vector<vector<double>> y= {{5,8,11,14,17,20,23,26,29,32,35,38}};
	y=transform(y);
	vector<vector<double>> w;
	w=LinearRegression(x,y,alpha,epoch);
}


