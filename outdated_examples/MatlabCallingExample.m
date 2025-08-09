% Created on Mon Oct 26
% This serve as an example how to call Spin wave toolkit from Matlab, but I
% think it is way more convinient to use e.g. Spyder
% If you do not now how to call Python from Matlab start with 
% https://www.mathworks.com/help/matlab/matlab_external/create-object-from-python-class.html
% @author: Ondrej Wojewoda

pathToSWT = fileparts(which('SpinWaveToolkit.py'));
if count(py.sys.path,pathToSWT) == 0
    insert(py.sys.path,int32(0),pathToSWT);
end

kxi = py.numpy.linspace(1, 25e6, int32(150));

% Dispersion of two first thickness modes of 30 nm NiFe thin film
material = py.SpinWaveToolkit.NiFe();
theta = pi/2;
phi = pi/2;
weff = 3e-6;
boundaryCond = 1;
Bext = 0.2;
d = 30e-9;

NiFeChar = py.SpinWaveToolkit.DispersionCharacteristic(Bext, material, d, kxi, theta, phi, weff, boundaryCond);
f00 =  double(NiFeChar.GetDispersion(0))*1e-9/(2*pi);
f11 =  double(NiFeChar.GetDispersion(1))*1e-9/(2*pi);

figure('name', 'Dispersion relation')
plot(kxi*1e-6, f00, kxi*1e-6, f11);
xlabel('kxi (rad/um)');
ylabel('Frequency (GHz)');
legend('00', '11')
title('Dispersion relation of NiFe n=0,1')

% Degeneration of the 00 and 22 mode in CoFeB
material = py.SpinWaveToolkit.CoFeB();
theta = pi/2;
phi = pi/2;
weff = 3e-6;
boundaryCond = 1;
Bext = 1.5;
d = 100e-9;

CoFeBchar = py.SpinWaveToolkit.DispersionCharacteristic(Bext, material, d, kxi, theta, phi, weff, boundaryCond);
w = CoFeBchar.GetSecondPerturbation(0, 2);
f02 = double(w{1,1}.data)*1e-9/(2*pi);
f20 = double(w{1,2}.data)*1e-9/(2*pi);

f00 = double(CoFeBchar.GetDispersion(0))*1e-9/(2*pi);
f22 = double(CoFeBchar.GetDispersion(2))*1e-9/(2*pi);

figure('name', 'Dispersion relation')
plot(kxi*1e-6, f02, kxi*1e-6, f20, kxi*1e-6, f00, kxi*1e-6, f22);
xlabel('kxi (rad/um)');
ylabel('Frequency (GHz)');
legend('02', '20', '00', '22')
title('Degeneration of the modes 00 and 22')