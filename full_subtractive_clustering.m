%**********************************************************************%
%               Substractive clustering algorithm                      %
%           Published by Stephen L Chiu                                %
%           Coded by Jorge cote                                        %
%               Arguments:                                             %
%                   * input: dataset input                             %
%                   * output: dataset output                           %
%                   * ra    : cluster ratio                            %
%                   * eps_over: superior limit cluster (0.5)           %
%                   * eps_sub : inferior limt cluster  (0.15)          %
%**********************************************************************%

function [xk_ast, pk_ast, sigma]= full_subtractive_clustering(input,output,ra,eps_over,eps_sub) 
x=[input' output'];      
minX = min(x);
maxX = max(x);
x=normaliz(x','maxmin')';
minX = min(x);
maxX = max(x);
datos=size(input,2);            %# muestras
k=1;
stop=0;
retest=0;
pi=zeros(datos,1);
alpha=4/(ra^2);     %parametro calculo inicial de potenciales para cada punto
rb=1.25*ra;     %sugerido en multiples trabajos
beta=4/(rb^2);      
%***************Compute first potential for each point********************%
for i=1:datos
    dis=0;
    for j=1:datos
       dis=dis+exp(-alpha*norm(x(i,:)-x(j,:))^2);   %calculo sumatoria Pi
    end
    pi(i)=dis;
end
%******************First cluster center and potential********************%
[maxpi,indx]=max(pi);     %maximo potencial
[pk_ast(1,:),indx1]=max(pi);     %maximo potencial (primer cluster)
xk_ast(1,:)=x(indx1,:);          %punto con el máximo potencial

while(stop==0)

   %**********************Potential revision***************************%
        
        for i=1:datos
        pi(i,:)=pi(i,:)-(maxpi*exp(-beta*norm(x(i,:)-x(indx,:)).^2));
        end
        [maxpi,indx]=max(pi);       %cluster candidate
%end
 
if(maxpi>(eps_over*pk_ast(1,:)))
    
      %*********************kth cluster center and potential***********%
       k=k+1;
    [pk_ast(k,1),indx]=max(pi);
    xk_ast(k,:)=x(indx,:);
   stop=0;
   retest=0;
elseif(maxpi<(eps_sub*pk_ast(1,:)))
    stop=1; 
else
    for i=1:size(pk_ast)
    d=x(indx,:)-xk_ast(i,:);
    abs_d=d*d';
    qd_dist(i,:)=abs_d;
    end
   dmin=sqrt(min(qd_dist));
    
    if(((dmin/ra)+(maxpi/pk_ast(1,:)))>=1)
        k=k+1;
        [pk_ast(k,1),indx]=max(pi);
        xk_ast(k,:)=x(indx,:);
        stop=0;
        
    else
       pi(indx)=0;
       [maxpi,indx]=max(pi);
        stop=0;        
    end       
end 
end
 %tomado de algoritmo substracting clustering anfis toolbox matlab
sigma = (ra* (maxX - minX)) / sqrt(8.0); %compute membership function width
end

















