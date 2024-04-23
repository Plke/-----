#include<iostream>
#include<string.h>
#include<list>
#include<algorithm>
#include<fstream>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include <random>
#include<ctime>
#define exp 2.718281828
#define net_num 3		//神经网络层数 
using namespace std;

class net
{
	public:
		vector<vector<double>> test_x;
		vector<vector<double>> test_y;
		vector<vector<double>> train_x;
		vector<vector<double>> train_y;

		int L;			//网络层数

		vector<int> net_ns;					//网络每层的数
		vector<vector<vector<double>>> w;		//每层神经网络之间的参数
		vector<vector<vector<double>>> net_bs;
		vector<string> act_funcs;
		vector<string> df_cost_func;
};

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
vector<vector<double> > Mult(vector<vector<double>> a,vector<vector<double>> b)			//矩阵对应相乘
{
	int n1=a.size();
	int n2=a[0].size();
	int n3=b.size();
	int n4=b[0].size();

	vector<vector<double>> res(n1,vector<double>(n4,0));

	if(n1!=n3||n2!=n4)
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
				res[i][j]=a[i][j]*b[i][j];
			}
		}
	}
	return res;
}
vector<vector<double> > Matrixadd(vector<vector<double>> a,vector<vector<double>> b)
{
	int n1=a.size();
	int n2=a[0].size();
	int n3=b.size();
	int n4=b[0].size();

	vector<vector<double>> res(n1,vector<double>(n2,0));

	if(n1!=n3)
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
				res[i][j]=a[i][j]+b[i][0];
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

vector<vector<double> > linear(vector<vector<double>> z)
{
	return z;
}

vector<vector<double> > dflinear(vector<vector<double>> z)
{
	vector<vector<double>> temp(z.size(),vector<double>(z[0].size(),1));
	return temp;
}

vector<vector<double> > sigmoidhelp(vector<vector<double>> z)
{
	for(int i=0; i<z.size(); i++)
	{
		for(int j=0; j<z[i].size(); j++)
		{
			z[i][j]=pow(exp,-z[i][j]);		//pow(x,y)	是x的y次方
		}
	}
	return z;
}

vector<vector<double> > sigmoid(vector<vector<double>> z)
{
	vector<vector<double> > temp(sigmoidhelp(z));
	for(int i=0; i<temp.size(); i++)
	{
		for(int j=0; j<temp[0].size(); j++)
		{
			temp[i][j]=1/(1+temp[i][j]);
		}
	}
	return temp;
}

vector<vector<double> > dfsigmoid(vector<vector<double>> z)
{
	vector<vector<double> > temp(sigmoid(z));
	for(int i=0; i<temp.size(); i++)
	{
		for(int j=0; j<temp[0].size(); j++)
		{
			temp[i][j]=(1-temp[i][j]);
		}
	}
	return MatrixMult(sigmoid(z) , temp);
}

double	mse(vector<vector<double>> a,vector<vector<double>> y)
{
	int num=y.size();		//数据的个数
	double res=0;
	for(int i=0; i<a.size(); i++)
	{
		for(int j=0; j<a[i].size(); j++)
		{
			res+=(a[i][j]-y[i][j])*(a[i][j]-y[i][j]);
		}
	}
	return res/num;
}

vector<vector<double> >	dfmse(vector<vector<double>> a,vector<vector<double>> y)
{
	int num=y.size();		//数据的个数
	double res=0;
	vector<vector<double>> temp(a.size(),vector<double>(a[0].size()));
	for(int i=0; i<a.size(); i++)
	{
		for(int j=0; j<a[i].size(); j++)
		{
			temp[i][j]=(a[i][j]-y[i][j])/num;
		}
	}
	return  temp;
}

vector<vector<double> > read_data(string data_name)
{
	fstream f;
	char name[30];
	strcpy(name,data_name.c_str());

	f.open(name,ios::in);
	int temp1=0,temp2=0;
	f>>temp1;		//行
	f>>temp2;		//列
	vector<vector<double>> tt(temp1,vector<double>(temp2));
	for(int i=0; i<tt.size(); i++)
	{
		for(int j=0; j<tt[i].size(); j++)
		{
			f>>tt[i][j];
		}
	}
	return tt;
}

vector<vector<vector<double> > > creat_w(vector<int> net_ns)
{
	vector<vector<vector<double>>> tempw(net_ns.size()-1);
	for(int i=0; i<net_ns.size()-1; i++)
	{
		vector<vector<double>> temp(net_ns[i+1],vector<double>(net_ns[i]));
		for(int row=0; row<temp.size(); row++)
		{
			for(int col=0; col<temp[0].size(); col++)
			{
				temp[row][col]=(rand()%10)/10.0;
			}
		}
		tempw[i]=temp;
	}
	return tempw;
}

vector<vector<vector<double> > > creat_bs(vector<int> net_ns)
{
	vector<vector<vector<double>>> tempw(net_ns.size()-1);
	for(int i=0; i<net_ns.size()-1; i++)
	{
		vector<vector<double>> temp(net_ns[i+1],vector<double>(1));
		tempw[i]=temp;
	}
	return tempw;
}

vector<vector<vector<double> > >  zeros_like(vector<vector<vector<double>>> a)
{
	vector<vector<vector<double>>> temp(a);
	for(int i=0; i<a.size(); i++)
	{
		for(int j=0; j<a[i].size(); j++)
		{
			for(int k=0; k<a[i][j].size(); k++)
			{
				temp[i][j][k]=0;
			}
		}
	}
	return temp;
}

vector<vector<vector<vector<double> > > >  forward(vector<vector<double>> x, net *mynet)
{
	int m = x[0].size();
	vector<vector<vector<double>>> z(mynet->net_ns.size());
	for(int i=0; i<mynet->net_ns.size(); i++)
	{
		vector<vector<double>> temp(mynet->net_ns[i],vector<double>(m,0));
		z[i]=temp;
	}
	vector<vector<vector<double>>> a(zeros_like(z));
	a[0]=x;
	for(int i=0; i<mynet->net_ns.size()-1; i++)
	{
		z[i+1]=Matrixadd(MatrixMult(mynet->w[i],a[i]), mynet->net_bs[i]);
		if(mynet->act_funcs[i+1]=="sigmoid")
		{
			a[i+1]=sigmoid(z[i+1]);
		}
		else if(mynet->act_funcs[i+1]=="linear")
		{
			a[i+1]=linear(z[i+1]);
		}
		else
		{
			cout<<"错误:激活函数名称错误"<<endl;
			system( "PAUSE ");
		}

	}
	return {a,z};
}

vector<vector<vector<vector<double> > > > backward(vector<vector<double>> y,vector<vector<vector<double>>> z,
        											vector<vector<vector<double>>> a,net *mynet)
{
	int l=mynet->net_ns.size();
	vector<vector<vector<double>>> delta=zeros_like(a);


	vector<vector<vector<double>>> grad_Ws=zeros_like(mynet->w);
	vector<vector<vector<double>>> grad_Bs=zeros_like(mynet->net_bs);
	if(mynet->df_cost_func[mynet->df_cost_func.size()-1]=="dfsigmoid")
	{
		delta[delta.size()-1]=Mult(dfmse(a[a.size()-1],y),dfsigmoid(z[z.size()-1]));
	}
	else if(mynet->df_cost_func[mynet->df_cost_func.size()-1]=="dflinear")
	{
		delta[delta.size()-1]=Mult(dfmse(a[a.size()-1],y),dflinear(z[z.size()-1]));
	}
	else
	{
		cout<<"错误:激活函数导数名称错误"<<endl;
		system( "PAUSE ");
	}
	for(int i=l-2; i>0; i--)
	{
		if(mynet->df_cost_func[mynet->df_cost_func.size()-1]=="dfsigmoid")
		{
			delta[i] = Mult(MatrixMult(transform(mynet->w[i]), delta[i + 1]) , dfsigmoid(z[i]));
		}
		else if(mynet->df_cost_func[mynet->df_cost_func.size()-1]=="dflinear")
		{
			delta[i] = Mult(MatrixMult(transform(mynet->w[i]), delta[i + 1]) , dflinear(z[i]));
		}
		else
		{
			cout<<"错误:激活函数导数名称错误"<<endl;
			system( "PAUSE ");
		}
	}

	for(int i=0; i<l-1; i++)
	{
		grad_Ws[i] = MatrixMult(delta[i + 1], transform(a[i]));
		for(int j=0; j<delta[i+1].size(); j++)		//每行相加
		{
			double temp=0;
			for(int k=0; k<delta[i+1][0].size(); k++)
			{
				temp+=delta[i+1][j][k];
			}
			grad_Bs[i][j][0]=temp;
		}
	}
	vector<vector<vector<double>>> (delta).swap(delta);
	return {grad_Ws, grad_Bs};
}
vector<vector<double> > grad_decent(net *mynet,double alpha=1,int max_epochs=5000,int  batch_size=32, int display_period=100)
{
	vector<double> train_costs ,test_costs ;
	for(int k=0; k<max_epochs; k++)
	{
		cout<<"次数 ："<<k+1<<endl;
		vector<int> sample_idxs(mynet->train_x[0].size());
		for(int i=0; i<mynet->train_x[0].size(); i++)
		{
			sample_idxs[i]=i;
		}
		random_shuffle(sample_idxs.begin(),sample_idxs.end());
		int num_batch=mynet->train_x[0].size()/batch_size;
		double train_cost=0;
		cout<<"num_batch : "<<num_batch<<endl;
		for(int batch_idx=0; batch_idx<num_batch; batch_idx++)
//		for(int batch_idx=0; batch_idx<5; batch_idx++)
		{
			if(batch_idx%100==1)
				cout<<"batch_idx : "<<batch_idx<<endl;
			vector<vector<double>> x(mynet->net_ns[0]),y(mynet->net_ns[mynet->net_ns.size()-1]);
			for(int col=batch_size * batch_idx; col<min(batch_size * (batch_idx + 1), int(mynet->train_x[0].size())); col++)
			{
				for(int row=0; row<mynet->train_x.size(); row++)
				{
					x[row].push_back(mynet->train_x[row][col]);
				}
				for(int row=0; row<mynet->train_y.size(); row++)
				{
					y[row].push_back(mynet->train_y[row][col]);
				}
			}
			vector<vector<vector<vector<double>>>> temp;
			temp=forward( x,mynet);
			vector<vector<vector<double>>> a,z;
			a=temp[0],z=temp[1];
			train_cost = train_cost + mse(a[a.size()-1], y) * x[0].size();
			temp = backward(  y, z, a, mynet);
			vector<vector<vector<double>>> grad_Ws,grad_Bs;
			grad_Ws=temp[0], grad_Bs=temp[1];
			vector<vector<vector<vector<double>>>>(temp).swap(temp);
			for(int i=0; i<mynet->net_ns.size()-1; i++)
			{
				for(int row=0; row<mynet->w[i].size(); row++)
				{
					for(int col=0; col<mynet->w[i][row].size(); col++)
					{
						mynet->w[i][row][col] = mynet->w[i][row][col] - alpha * grad_Ws[i][row][col];
					}
					for(int col=0; col<mynet->net_bs[i][row].size(); col++)
					{
						mynet->net_bs[i][row][col] = mynet->net_bs[i][row][col] - alpha * grad_Bs[i][row][col];
					}
				}
			}
			vector<vector<double>> (x).swap(x);
			vector<vector<double>> (x).swap(y);
		}

		train_cost = train_cost / mynet->train_x[0].size();
		if (k % display_period == 0)
		{
			train_costs.push_back(train_cost);
			sample_idxs.clear();
			sample_idxs.resize(mynet->test_x[0].size());
			for(int i=0; i<mynet->test_x[0].size(); i++)
			{
				sample_idxs[i]=i;
			}
			random_shuffle(sample_idxs.begin(),sample_idxs.end());
			vector<int> (sample_idxs).swap(sample_idxs);
			double test_cost = 0;
			num_batch=mynet->test_x[0].size()/batch_size;
			cout<<"num_batch : "<<num_batch<<endl;
			for(int batch_idx=0; batch_idx<num_batch; batch_idx++)
//			for(int batch_idx=0; batch_idx<5; batch_idx++)
			{
				if(batch_idx%100==1)
					cout<<"batch_idx : "<<batch_idx<<endl;
				vector<vector<double>> x(mynet->net_ns[0]),y(mynet->net_ns[mynet->net_ns.size()-1]);
				for(int col=batch_size * batch_idx; col<min(batch_size * (batch_idx + 1), int(mynet->test_x[0].size())); col++)
				{
					for(int row=0; row<mynet->test_x.size(); row++)
					{
						x[row].push_back(mynet->test_x[row][col]);
					}
					for(int row=0; row<mynet->test_y.size(); row++)
					{
						y[row].push_back(mynet->test_y[row][col]);
					}
				}
				vector<vector<vector<vector<double>>>> temp;
				temp = forward(x,mynet);
				vector<vector<vector<double>>> a,z;
				a=temp[0],z=temp[1];
				vector<vector<vector<vector<double>>>>(temp).swap(temp);
				test_cost = test_cost + mse(a[a.size()-1], y) * x[0].size();
				vector<vector<double>> (x).swap(x);
				vector<vector<double>> (x).swap(y);
			}
			test_cost = test_cost / mynet->test_x[0].size();
			test_costs.push_back(test_cost);
			cout<<"epoch = "<<k<<"     ,train_cost=" <<train_cost<<"     ,test_cost="<<test_cost<<endl;
		}
	}
	return {train_costs, test_costs};
}


int main()
{
	srand((unsigned int)(time(NULL)));
	net *mynet =new net;
	mynet->test_x=read_data("test_x.txt");
	mynet->test_y=read_data("test_y.txt");
	mynet->train_x=read_data("train_x.txt");
	mynet->train_y=read_data("train_y.txt");
	cout<<"train_x shape : "<<mynet->train_x.size()<<" "<<mynet->train_x[0].size()<<endl;
	cout<<"train_y shape : "<<mynet->train_y.size()<<" "<<mynet->train_y[0].size()<<endl;
	cout<<"read data over"<<endl;
	int x_dim= mynet->test_x.size();					//每个数据是一个列向量
	int y_dim= mynet->test_y.size();

	mynet-> net_ns= {x_dim,1024,y_dim};					//网络每层的数
	mynet-> w=creat_w(mynet->net_ns);		//每层神经网络之间的参数
	mynet-> net_bs=creat_bs(mynet->net_ns);
	mynet-> act_funcs= {"no","sigmoid","linear"};
	mynet-> df_cost_func= {"no","dfsigmoid","dflinear"};

	int max_epochs = 100;
	int display_period = max_epochs / 100;
	int batch_size = 32;
	double alpha = 0.05;
	grad_decent(mynet,alpha,max_epochs,batch_size,display_period);
	return 0;
}

