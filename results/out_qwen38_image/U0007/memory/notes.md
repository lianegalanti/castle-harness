Task: 19-part table-like structure. All upright yaw0. Tolerance 5mm.

Part sizes [x,y,z]:
0,1,2,3 = 0.02,0.02,0.05 (corner posts)
4,5 = 0.14,0.02,0.01 (long slabs in x)
6,7,9 = 0.02,0.1,0.01 ; 8,10 = 0.02,0.1,0.02 (slabs in y)
11 = 0.02,0.03,0.01
12 = 0.06,0.12,0.01 (big cross slab)
13 = 0.06,0.04,0.02
14 = 0.02,0.02,0.03
15 = 0.03,0.01,0.02
16 = 0.02,0.01,0.05 (tall)
17 = 0.03,0.01,0.03
18 = 0.02,0.02,0.02

Views: TOP(x horiz,y vert) FRONT(x horiz,z vert) SIDE(y horiz,z vert).
Table z=0. part on table: z=size_z/2.

BASE HYPOTHESIS (build order):
posts z=0.025:
0=(0.345,-0.025) 1=(0.475,-0.025) 2=(0.345,0.065) 3=(0.475,0.065)
long slabs z=0.055 on posts:
4=(0.405,-0.025) 5=(0.405,0.065)
cross slab 12 z=0.065 on 4&5: 12=(0.41,-0.03)
UPPER (z>=0.07): 6,7,8,9,10,11,13,14,15,16,17,18 - TBD by views + planner.

Planner validates support: use it to fix z levels. observe after batches.
