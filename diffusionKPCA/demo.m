clear all;
close all;  
clc;

AF = 2;
CenSize = 10;
snr = 20;
PC = 10;
TH = 6;

addpath(pwd,'KernelLib' );
addpath(pwd, 'ReconLib' );
addpath(pwd, 'PreImLib' );
addpath(pwd, 'FFTLib' );
addpath('/cbil1/czhang46/Data');

load W4_rawdata

initialize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
itrNum = zeros(1,ImgSize(3));
disp( '1 -- Initialization is DONE!' );

disp( '==============================================================' );
for xdim = 1:ImgSize(3)
    clear KerPara
disp( strcat('2 -- Kernel Eigen Decomposition on Training ......',num2str(xdim)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TrainingDataKPCADTI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp( KernelMode, 'Gauss' )
    KerPara.Mode   = 'Gauss';
    KerPara.c      = GaussSigma;
    KerPara.d      = NaN;
elseif strcmp( KernelMode, 'Poly' )
    KerPara.Mode   = 'Poly';
    KerPara.c      = PolyConst;
    KerPara.d      = PolyPower;
    KerPara.STh     = STh;
    KerPara.SThDec  = SThDec;
else
end

KerPara.ItrNum  = KPCAItrNum;
KerPara.RegPara = KPCARegVal;
KerPara.ALMPara = ALMRegVal;
KerPara.LamdaReg= LamdaReg;

PreData =  squeeze(abs(LowSos(:,:,xdim,:)));
ReconSize = size(PreData);
MSEVec  = zeros( 1, MaxItrNum );

B = PreData;%zeros( ImgSize );  %%% Comment by UN This is LowResImg
U = zeros(ReconSize);
TraData = PreData;
TraData = permute( TraData, [ 3 1 2 ] );
TraData = reshape( TraData, ReconSize(3), ReconSize(1)*ReconSize(2));
RandIdx = randperm(ReconSize(1)*ReconSize(2), KPCATraNum );
TraData = TraData(:,RandIdx );

Timer1=tic;
% ------ KPCA for training data ------
[ ~, Alpha, D, Kxx, Kxc, ~, ~ ] = KPCA( TraData, TraData, KerPara );
KEigTime=toc(Timer1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp( '2 -- Kernel Eigen Decomposition on Training is DONE!' );
disp( '==============================================================' );
MSEMat=zeros(length(PC),length(TH));
TimerMat=zeros(length(PC),length(TH));
KPCAPctNum=PC;
SThDec=TH;
KerPara.SThDec  = SThDec;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ReconDTI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Alpha1= Alpha( :, 1 : KPCAPctNum );
D1= D( 1 : KPCAPctNum );
% ------ projectino for test data, compute Gamma ------
KerPara.Alpha = Alpha1;
KerPara.D     = D1;      
KerPara.Kxx   = Kxx;
KerPara.Kxc   = Kxc;  
MSEVec=zeros(1,500);

clear CurDataPrev;
Timer2 = tic;
Mask = squeeze(Mask3D(:,:,xdim,:,:));
CurData=abs(IFFT2_4D(squeeze(sampledKData(:,:,xdim,:,:)), Mask, 1 ));
ReconSize = size(squeeze(sos(CurData,3)));
RefChanData = squeeze(sos(abs(RefData(:,:,xdim,:,:)),4));
KData = squeeze(sampledKData(:,:,xdim,:,:));
for ItrIndex = 1 : 500 % MaxItrNum   
   %  fprintf( 'Main iteration: %d ......\n', ItrIndex );

%% ==========================  Kernel subproblem ( Gamma and x-subproblem ) ==========================
    BlkData = squeeze(sos(abs(CurData),3));
    TpfData = permute( BlkData, [ 3 1 2 ] );
    TpfData = reshape( TpfData, ReconSize(3), (ReconSize(1)*ReconSize(2)) );   
    [ PrjCoff, ~, ~, ~, ~, Kxy, Kyc ] = KPCA( TpfData, TraData, KerPara );
    % Using Sth like in ktSense
    MaxPjC=max(abs(PrjCoff(:)));
    minPrjCoff=min(abs(PrjCoff(:)));
    %PrjCoff=(abs(PrjCoff)-KerPara.SThDec).*PrjCoff./abs(PrjCoff).*(abs(PrjCoff)>KerPara.SThDec);
    PrjCoff= wthresh(PrjCoff,'h',KerPara.SThDec);
    

    Gamma = KerPara.Alpha * PrjCoff;
    Gamma = Gamma - repmat( mean( Gamma, 1 ), KPCATraNum, 1 ) + 1 / KPCATraNum;
    CurDataPrev=sos(CurData,3);
    %% PreImage Problem. %% Subthresholding is used there within 

    PreIm=FastPreImPol(TraData,Gamma,KerPara);  
    PreIm=reshape(PreIm, ReconSize(3), ReconSize(1), ReconSize(2));
    PreIm=permute(PreIm,[2,3,1]);
    PreIm=abs(PreIm);
     MaxPreIm=max(PreIm(:));
     PreIm=PreIm./max(PreIm(:))*max(abs(CurDataPrev(:)));


    %  CurDataPrev=sos(CurData,3); % This is used to calculate the MSE 
    UpdateImg = zeros(ImgSize(1),ImgSize(2),ImgSize(4),ImgSize(5));
      for frame = 1:ImgSize(5)
          for coil = 1:ImgSize(4)
              UpdateImg(:,:,coil,frame) = PreIm(:,:,frame).*LowSen(:,:,xdim,coil,frame);
          end
      end
      
      UpData = FFT2_4D( UpdateImg, ones( ImgSize(1), ImgSize(2),ImgSize(4),ImgSize(5)), 1 );
      Y=UpData;
      Y = Y.*(Mask==0) + KData; 
      CurData=IFFT2_4D( Y, ones( ImgSize(1), ImgSize(2),ImgSize(4),ImgSize(5)), 1 );
      CurData(isnan(CurData))=0;
 
  
   
   ErrData = abs(RefChanData)-abs(CurDataPrev);
   MSEVec(ItrIndex) = mean( abs( ErrData(:) ).^2 );
   aa=MSEVec(ItrIndex);

    temp = abs(squeeze(sos(abs(CurData),3))-abs(CurDataPrev));
    DiffCur=norm(temp(:));
    if(ItrIndex >2)
        if(MSEVec(ItrIndex-1)< MSEVec(ItrIndex)|| DiffCur<2.8e-15)
            aa=MSEVec(ItrIndex-1);           
            break;
        end
    end
end%%%  End MaxItrNum


PrjTime=toc(Timer2);
ItrTime=PrjTime;
reconImg(:,:,:,:,xdim) = abs(CurData);
end

reconImg = permute(reconImg,[5,1,2,3,4]);
recon = permute(squeeze(sos(NDWI,4)),[3,1,2,4,5]);
recon(:,:,:,3:(ImgSize(5)+2)) = squeeze(sos(abs(reconImg),4));

filename = strcat('W4__Recon3D_AF',num2str(AF));
save(filename,'recon','AF');