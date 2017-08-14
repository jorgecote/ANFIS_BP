%*********************************ANFIS********************************%
%            Algoritmo de inferencia difusa ANFIS                      %
%              con consecuente tipo Takagi-sugeno entrenado por        %
%              algoritmo de retro-propagación                          %
%       Argumentos:                                                    %
%           * dataset    : Conjunto de datos de entrenamiento          %
%           * dataset_val: Conjunto de datos de validación             %
%           * epochs     : Cantidad máxima de iteraciones              %
%           * ra         : Radio de clustering                         %
%           * n          : Tasa de aprendizaje n (eta)                 %
%**********************************************************************%

function [MF,output2,traindataset, O5,O5_val, etotal,etotal_val, e  ]=anfis_algo(dataset,dataset_val,epochs,ra,n)
%tamaño de conjunto de datos
traindataset=dataset;
datasetsize=size(traindataset);
Q=datasetsize(1);
input=traindataset(:,1:end-1)';
output=traindataset(:,end)';
input_val=dataset_val(:,1:end-1)';
%**********Función auxiliar para clustering substractivo*****************%
[xk_ast, pk_ast, sigma1]= full_subtractive_clustering(input,output,ra,0.5,0.15);
%Normalización de conjuntos de datos
 input=normaliz(input,'maxmin');
 input_val=normaliz(input_val,'maxmin');
datos=size(input,2);
variables=size(input,1);
%Cantidad de reglas
MF=size(pk_ast,1);
%**************Ancho de funciones de pertenencia******************%
%tomado de extracting fuzzy rules from data (chiu)
for i=1:MF
sigma(i,:)=sigma1;
end
%inicialization
O1=zeros(MF,variables);
O2=zeros(MF,variables);	
O3=zeros(MF,datos);
dO2_dO1=zeros(MF,variables,datos);
dE_dc=zeros(MF,variables,datos);
dE_dsigma=zeros(MF,variables,datos);
etotal=zeros(1,epochs);
n1=n;
n2=n;
iter=0;
etotal=1;

for iter=1:epochs
%Inicializacion derivadas del error
     dE_dc=zeros(MF,variables,datos);
dE_dsigma=zeros(MF,variables,datos);
%***************************
    for t=1:datos
%========================================================================%
%****************************Fordward Pass*****************************%
%========================================================================%
%Calculo de la capa 1
        for i=1:MF   %MFxvariablesxdata
    O1(i,:,t)=exp(-0.5*(((input(:,t))'-xk_ast(i,1:end-1))./sigma(i,1:end-1)).^2);
        end         
    end
    %Calculo de la capa 2
    O2=prod(O1,2);    %MFx1xdata
    O2=reshape(O2,MF,datos);  %MFxdata
   %Calculo de la capa 3
    for i=1:MF
    O3(i,:)=O2(i,:)./sum(O2);    %MFxdata
    end
      %Calculo de la capa 4    
    A_=[input'];  
    for t=1:datos
        for i=1:MF
        A(i,:,t)=O3(i,t).*A_(t,:);         
        end
    %Vector de parámetros del consecuente
    temp2(t,:)=reshape(A(:,:,t)',1,(variables)*MF);  %vector de parametros
    end
%Cálculo de mínimos cuadrados
X=temp2\output';
    
    %Sumatoria capa 5
    O5=temp2*X;
    %Error cuadrático
 e=(output'-O5).^2;
 %Error MSEde aprendizaje
 etotal(iter)=(1/datos)*sum(e);       %error total por iteracion
etotal(iter)
% Salida de validación
O5_val=anfis_feedfwd(input_val,xk_ast,sigma,X,variables,MF);
%Error MSE de validación
etotal_val(iter)=(1/size(dataset_val,1))*sum((dataset_val(:,end)-O5_val).^2); 
%========================================================================%
%****************************Retro-porpagación***************************%
%========================================================================%
%Calculo de las derivadas
    dE_dO5=-(2/datos)*(output'-O5);             %dataX1
    X_=reshape(X,variables,MF);
    dO5_dO3=(A_*X_);   %dataxMF
     for i=1:MF
            dO3_dO2(i,:)=(sum(O2)-O2(i,:))./(sum(O2)).^2;  %MFxdata
     end
         for j=1:variables
            temp=O1;
            temp(:,j,:)=[];
            dO2_dO1(:,j,:)=prod(temp,2);   %MFxvariablesxdatos
         end  
        %dO2_dO1=reshape(dO2_dO1,MF,datos);
        for t=1:datos
        for i=1:MF
            d_dc(i,:,t)=(((input(:,t))'-xk_ast(i,1:end-1))./sigma(i,1:end-1).^2);
            d_dsigma(i,:,t)=((((input(:,t))'-xk_ast(i,1:end-1)).^2)./sigma(i,1:end-1).^3);
        end
        end
        dO1_dc=d_dc.*O1;    %MFxvariablesxdata
        dO1_dsigma=d_dsigma.*O1;      %MFxvariablesxdata
        for j=1:variables
            for i=1:MF
            dE_dc(i,j,:)=dE_dO5.*dO5_dO3(:,i).*dO3_dO2(i,:)'.*reshape(dO2_dO1(i,j,:),datos,1).*reshape(dO1_dc(i,j,:),datos,1);
            
            dE_dsigma(i,j,:)=dE_dO5.*dO5_dO3(:,i).*dO3_dO2(i,:)'.*reshape(dO2_dO1(i,j,:),datos,1).*reshape(dO1_dsigma(i,j,:),datos,1);
            end
        end
        dE_dc=sum(dE_dc,3);
        dE_dsigma=sum(dE_dsigma,3);
        %Actualización de parámetros
        xk_ast(:,1:variables)=xk_ast(:,1:variables)-(n1*dE_dc);
        sigma(:,1:variables)=sigma(:,1:variables)-(n2*dE_dsigma);
        %========================================================================%
%******************** Fin iteración**************************************%
%========================================================================%
 
        end

 output2=X_;
end











