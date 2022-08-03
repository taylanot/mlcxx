// Gmsh project created on Mon Sep 14 09:33:57 2015
a = 1.;
R = 0.2;
t = Pi/3;
b = Sin(t)*a;
bR = Sin(t)*R;
c = a*Cos(t);
cR = R*Cos(t);

d = 1.;

nx = 80;
nR = R*nx;
ns = nx-2*nR;

Point(1) = {0, 0, 0, d};
Point(2) = {a, 0, 0, d};
Point(3) = {a+c, b, 0, d};
Point(4) = {c, b, 0, d};
Point(5) = {R,0,0,d};
Point(6) = {a-R,0,0,d};
Point(7) = {a+cR,bR,0,d};
Point(8) = {a+c-cR,b-bR,0,d};
Point(9) = {a+c-R,b,0,d};
Point(10) = {c+R,b,0,d};
Point(11) = {c-cR,b-bR,0,d};
Point(12) = {cR,bR,0,d};

Line(1) = {1, 5};
Line(2) = {5, 6};
Line(3) = {6, 2};
Line(4) = {2, 7};
Line(5) = {7, 8};
Line(6) = {8, 3};
Line(7) = {3, 9};
Line(8) = {9, 10};
Line(9) = {10, 4};
Line(10) = {4, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};
Circle(13) = {5, 1, 12};
Circle(14) = {7, 2, 6};
Circle(15) = {9, 3, 8};
Circle(16) = {11, 4, 10};
Line Loop(17) = {13, -11, 16, -8, 15, -5, 14, -2};
Plane Surface(18) = {17};
Line Loop(19) = {1, 13, 12};
Plane Surface(20) = {19};
Line Loop(21) = {3, 4, 14};
Plane Surface(22) = {21};
Line Loop(23) = {7, 15, 6};
Plane Surface(24) = {23};
Line Loop(25) = {10, 16, 9};
Plane Surface(26) = {25};

Transfinite Line{1,3,4,6,7,9,10,12} = nR;
Transfinite Line{2,5,8,11} = ns;
Transfinite Line{14,16} = 2*nR;
Transfinite Line{13,15} = nR;

Periodic Line{1} = {9};
Periodic Line{2} = {8};
Periodic Line{3} = {7};
Periodic Line{4} = {12};
Periodic Line{5} = {11};
Periodic Line{6} = {10};

Physical Line(1) = {1,2,3};
Physical Line(2) = {4,5,6};
Physical Line(3) = {7,8,9};
Physical Line(4) = {10,11,12};

Physical Surface(0) = {18};
Physical Surface(1) = {20,22,24,26};

