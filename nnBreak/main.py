#!/usr/bin/python
#SBATCH --job-name=break_nn
#SBATCH --cpus-per-task=2
#SBATCH --output=out.txt
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --mem-per-cpu=8G


import sys
# sys.path.append('../..')
import os
sys.path.append("/cs/labs/guykatz/zivarda/Marabou/nnBreak")
sys.path.append("/cs/labs/guykatz/zivarda/Marabou")
from maraboupy import Marabou
from NNet.python.nnet import *
# nnet = NNet('ACASXU_experimental_v2a_1_1.nnet')
from NNet.utils.writeNNet import writeNNet
from maraboupy.MarabouUtils import *
from maraboupy import MarabouCore
import shutil
import multiprocessing

EPSILON = 0.1
INPUT_SIZE = 5

# 1.0950150884, 2.4163122489 ]
# x306: [ 0.5699494621, 1.0729019940 ]
# x307: [ -0.3926160224, 0.3225942331 ]
# x308: [ -20.2756940116, -0.1103267966 ]
# x309: [ -0.0623810127, 0.6098006692 ]
# x310: [ -20.9221051564, -0.3271184127 ]
# x311: [ -0.8751796535, 0.0707926613 ]
# x312: [ 0.9642503455, 1.7610700368 ]
# x313: [ -21.7601389190, -0.7686243968 ]
# x314: [ -34.3375238771, -2.5641075635 ]
# x315: [ -50.2799839728, -3.7616541924 ]
# x316: [ -0.5144353561, 0.4609269813 ]
# x317: [ -33.2233855873, -1.2711629348 ]
# x318: [ -0.2951024911, 0.7752778788 ]
# x319: [ 0.1363710236, 0.4056883057 ]
# x320: [ -1.7786192831, -0.2826376600 ]
# x321: [ -0.1475879448, 2.1902611715 ]
# x322: [ 0.3483684980, 1.3051937306 ]
# x323: [ 0.1736878674, 1.0436820273 ]
# x324: [ -20.8816826741, -0.9143443239 ]
# x325: [ -25.2728685944, -1.5928462592 ]
# x326: [ 6.7586399832, 10.7661305155 ]
# x327: [ -68.9716753515, -1.4413302546 ]
# x328: [ -0.9914150250, 0.0992900121 ]
# x329: [ -22.0229762627, -0.9025537576 ]
# x330: [ -70.1441942530, -3.0766911426 ]
# x331: [ -88.2285792845, -0.1025638903 ]
# x332: [ -1.4625888734, 0.2425199775 ]
# x333: [ 0.0438261420, 0.5003497821 ]
# x334: [ 6.2138617236, 7.6261003263 ]
# x335: [ -0.4328320474, 0.2763782523 ]
# x336: [ -16.6040424157, -0.0454262878 ]
# x337: [ -0.7124093451, 0.7254980309 ]
# x338: [ -14.6599895108, -0.7022542030 ]
# x339: [ -48.0791306046, -3.5411294142 ]
# x340: [ -0.4094044560, 0.3416291001 ]
# x341: [ -39.3534742496, -1.1117629726 ]
# x342: [ -26.9910814526, -0.3427847845 ]
# x343: [ -2.4119760519, -0.2652034484 ]
# x344: [ 0.0515534348, 0.7432108026 ]
# x345: [ -0.1411148316, 0.4461249981 ]
# x346: [ 0.2939406162, 1.0489079402 ]
# x347: [ -17.4204816443, -0.7708183305 ]
# x348: [ 0.3278334803, 1.7160655190 ]
# x349: [ -0.4324155761, 1.5187490676 ]
# x350: [ 0.6126967448, 0.8755548772 ]
# x351: [ -22.3505408066, -0.6522864021 ]
# x352: [ 0.1441523712, 1.1713635025 ]
# x353: [ 0.2638928881, 0.6918698217 ]
# x354: [ -19.2333133214, -3.7520042088 ]

layer3_max = [0.3579362343,-0.8814376611,0.1863698709,-1.5100138611,-0.0935008655,0.1122973343,0.1922972623,-0.3139951985,-0.7171807629,
0.1441987867,0.4457304467,0.2513349765,0.3272180552,0.3338135255,0.3627451043,0.3048520709,1.5690971941,-0.5923438221,0.0070562989,0.2712227895,
-0.7823034111,-1.2764514564,-0.0150831267,-1.1295004446,-0.2743789721,0.2008755970,-2.9418917538,-0.0386543964,0.5588116944,1.2461661134,
0.7046533862,-0.7233598616,-0.0135159014,0.1059394040,-1.2400325964,0.3506924359,-0.2115578291,0.3611104655,0.0906144381,-0.3291736457,
0.9629646407,-1.5943542769,-0.5991447694,1.0688480905,-0.3940793288,-1.5787686974,0.4668544054,-0.3390685448,6.8481678664,-0.6375154394]

layer3_min = [0.1860577001,-4.9360105468,-0.1185079612,-4.7705990347,-0.2817075151,-0.1458498818,-0.1014593333,-2.9105155948,-3.5495547181,
-0.6382354623,0.1237949055,0.0338948789,-0.1234206516,0.0579596966,-0.0779787894,0.0553366681,0.9628545291,-2.7705499530,-0.7364039970,
-0.2380033629,-3.9536509362,-5.2171259751,-2.7956870490,-4.8521303527,-2.6060417832,-0.1469855362,-8.2822803559,-1.6244555612,0.4143814865,
0.8136757325,0.5013649948,-2.8533118947,-0.3683981120,-0.4098208824,-4.9661597358,-0.5235238746,-1.9647539115,-0.0454459112,-0.2680730921,
-2.6041269256,0.2948460413,-4.0254444495,-3.0344756992,0.5320166598,-3.9592041801,-6.3825467262,0.2997278658,-3.8547971070,2.6295787262,
-3.0918847850]


layer4_min=[1.0950150884,0.5699494621, -0.3926160224,-20.2756940116,-0.0623810127,-20.9221051564,-0.8751796535,0.9642503455, -21.7601389190,-34.3375238771,
-50.2799839728,-0.5144353561,-33.2233855873,-0.2951024911,0.1363710236, -1.7786192831,-0.1475879448,0.3483684980, 0.1736878674, -20.8816826741,
-25.2728685944,6.7586399832, -68.9716753515,-0.9914150250,-22.0229762627,-70.1441942530,-88.2285792845,-1.4625888734,0.0438261420, 6.2138617236,
-0.4328320474,-16.6040424157,-0.7124093451,-14.6599895108,-48.0791306046,-0.4094044560,-39.3534742496,-26.9910814526,-2.4119760519,0.0515534348,
-0.1411148316,0.2939406162, -17.4204816443,0.3278334803, -0.4324155761,0.6126967448, -22.3505408066,0.1441523712, 0.2638928881, -19.2333133214]

layer4_max=[2.4163122489,1.0729019940,0.3225942331,-0.1103267966, 0.6098006692,-0.3271184127, 0.0707926613,1.7610700368,-0.7686243968,-2.5641075635,-3.7616541924, 0.4609269813,
-1.2711629348, 0.7752778788,0.4056883057,-0.2826376600, 2.1902611715,1.305193730,1.0436820273,-0.9143443239,-1.5928462592,10.7661305155,-1.4413302546,
 0.0992900121,-0.9025537576,-3.0766911426,-0.1025638903, 0.2425199775,0.5003497821,7.6261003263, 0.2763782523,-0.0454262878, 0.7254980309,-0.7022542030,
-3.5411294142,0.3416291001,-1.1117629726,-0.3427847845, -0.2652034484,0.7432108026, 0.4461249981,1.0489079402,-0.7708183305,1.7160655190,1.5187490676,# print("Num Inputs: %d"%nnet.num_inputs())
0.8755548772,-0.6522864021,1.1713635025,0.6918698217,-3.7520042088,]

layer6_min = [-1.2577900565, -0.9085347733, 0.7876611231, -3354.9183138878, -1949.2388207806, -0.1806404391, -2645.5834714900, -49.7228899846,
              -1306.4888446232, -11430.4209644146, -1218.7907036738, -12096.1358315762, -0.6576773826, -6952.5257818794, -21.2918871624, -0.6486369638,
              -37639.7974246451, -0.6303538993, -1.6308523930, -6664.5402170044, -40.3107286644, -32.5136251212,
              -4.2236205598, -0.0587020396, -3437.4920851626, -269.8997362723,
              -3.8869334034, -0.1061471772, -27024.4289784825, -20.6233009530,
              -1342.0724576574,
              -9325.7050468888, -18441.1056081861, - 6.5317279899,
              0.3447851023, -208.6769318286, -4325.6639497603, -1.2671412102,
              0.1546310541, -5.2331229742,
              -334.9658679874, -13316.2851639033, -6.4296651986,
              -1051.4357165181, -4.8163153834, -12.9296174329, 0.1539532604,
              -0.6903828597, -16.8628424331, -14.5603604843,]





layer6_max =[2.0011735206, 0.0839028735, 1.4135784152, -1.5801125099,  -3.4545554631,  0.7592852412,  -17.1257261917,  3.1766854279,  -6.6410208172,
 -2.0711597142, -6.4554422445,  -1.0033965546, 0.0567469200, -1.2970620050,2.9578539532, 1.5978756117, 1.6820809330, 0.6213849601, 0.0734063154,
             -1.0409812694,0.7874837356, 0.8132318400,
             0.6697546086, 0.7632690224, -16.1809552062, -0.4764677865,
             1.5833163940, 0.7234256015, -17.4390353553, 8.9711141376,
             -0.0742044551, -1.1114950235, -1.7311026738,
             0.4367006948, 0.9017164188, -0.0712969403, -20.4315808266,
             1.8998240058, 0.8968186916, 2.3834097274, -0.2552017750,
             -12.5210647576, 2.8276771947,
             -1.0619765430, 1.8469377219, 0.2276278132, 0.9864785277,
             0.2553204312, 3.5490399904, 3.7900288218]


layer5_max = [1.8926702833,1.6708462235,-0.6982667019,0.8825804504,1.0461302262,-0.0133605913,-0.1035923977,-5.2357514766,-1.0064824567,
-0.5577013310,0.2602723599,0.2357026637,0.0946585896,0.5800970845,-3.2969994213,-0.3754713396,-2.1763684323,-8.2278239986,-0.2241543269,
1.4887350428,-5.2761431087,0.1334209183,1.5617322208,-2.4224192132,3.8584816352,7.2099886046,-0.7577706994,0.7294172949,-0.3290899298,
-0.6821038908,0.3746757430,-0.4865102421,14.7873286626,-3.4192783752,-6.9328033324,0.9662560853,4.0343620210,0.4692113488,-0.6906602381,
-0.3420453009,3.0389384612,-3.4430355118,0.2356207916,0.5428152791,-1.4714946021,-5.1019649758,1.2078901940,2.1479651594,-0.1728296346,
0.2781996166,
]

layer5_min = [0.7857593555,1.0490208263,-220.6479813864, -1.5867641502,-1.0917872465,-138.7169500251, -102.7532523327, -1017.7355269807,
-24.9628095206,-471.9840251551, -0.5459755589,-1.6714097373,-1.8338645011,-1.1078336211,-162.3448163992, -142.8382273360, -216.2175254224,
-878.4248352683, -234.1091961919, 0.0157215491,-304.5750506468, -1.7184624269,-5.5989216422,-1361.4603700534,-1.0936430795,-4.1856540841,
-431.9237368909, -3.7341759398,-334.4071382452, -197.1381918649, -1.7452098842,-894.4082390654, 12.2628152457,-651.1039756348, -761.6175136533,
-1.3686898288,1.2267142217,-2.3098581745,-662.3366312190, -297.8803275796, -0.0344154268,-185.2351101504, -2.6779395631,-0.3166148691,
-344.5152483483, -515.5701586117, -2.2736359906,0.2242877314,-250.0997459316, -3.3613782639,]

layer5_max_nnet11 = [449.564,987.64,193.246,1883.34,294.632,274.651,437.747,679.953,542.704,274.9,284.414,440.46,648.017,2002.67,588.37,165.678,521.356,641.263,347.56,314.563,793.25,159.097,323.446,758.737,269.056,243.684,349.152,499.406,837.143,886.165,495.692,187.15,974.176,361.118,432.516,977.646,798.691,249.823,212.214,713.062,226.554,150.893,367.389,106.132,1769.05,553.85,213.663,570.396,572.154,776.833,]
layer5_min_nnet11 = [-478.196,-1430.58,-578.055,-2033.97,-710.384,-404.603,-327.813,-1019.61,-849.408,-240.332,-189.428,-963.707,-1500.63,-3443.23,-1490.24,-751.067,-2361.97,-598.405,-3053.18,-654.566,-1414.61,-250.864,-514.716,-1229.73,-553.851,-153.531,-301.406,-492.803,-2395.58,-2278.93,-1171.86,-182.656,-1130.94,-346.403,-1015.9,-3110.26,-2882.83,-480.067,-306.201,-1388.66,-480.416,-181.693,-1002.55,-790.978,-1740.48,-246.774,-238.398,-753.634,-2328.67,-609.893,]

# layer

# print("Num Outputs: %d"%nnet.num_outputs())
# print("One evaluation:")
# print(nnet.evaluate_network([15299.0,0.0,-3.1,600.0,500.0]))
# print("\nMultiple evaluations at once:")
# print(nnet.evaluate_network_multiple([[15299.0,0.0,-3.1,600.0,500.0],[15299.0,0.0,-3.1,600.0,1200.0]]))

UNSAT = "unsat"
SAT = "sat"
MID = "MID"
NOT_MID = "NOT_MID"

"""
Separate Files to 2 parts of nn:
1. data about nn: change number of input, output, max layer size.
2. layer sizes: separate this line to 2, the beginning in part1, the end in part2.
3. min value: part1: same as original nn, part2: line of 0
4. max value: part1: same as original nn, part2: line of max_val (which val?)
5. mean: part1: same as original nn with last val=0, part2: line of 0s with last val as the last in the original
6. range: part1: same as original nn with last val=1, part2: line of 1s with last val as the last in the original
7. weights: cut in the corresponding layer.
"""

def split_nn(path, nn_filename, layer_number):
    """
    Split nn to two nn at layer_number. Create two files with the splitted
    :param nn_filename: Neural network file name
    :param layer_number: number of layer to split on.
    :return: names of created files.
    """
    nn = NNet(path+"/"+nn_filename)
    mid_layer_size = nn.layerSizes[layer_number]

    weights1 = nn.weights[:layer_number]
    weights2 = nn.weights[layer_number:]
    biases1 = nn.biases[:layer_number]
    biases2 = nn.biases[layer_number:]

    mins1 = nn.mins
    # mins2 = [0]*mid_layer_size
    mins2 = [x if x>=0 else 0 for x in layer4_min]
    maxes1 = nn.maxes
    # maxes2 = [10E100]*mid_layer_size
    # maxes2 = [1000]*mid_layer_size
    maxes2 = [x if x>=0 else 0 for x in layer4_max]

    means1 = nn.means
    last_mean = nn.means[-1]
    means1[-1] = 0
    means2 = [0]*mid_layer_size+[last_mean]
    ranges1 = nn.ranges
    last_range = nn.ranges[-1]
    ranges1[-1] = 1
    ranges2 = [1]*mid_layer_size+[last_range]

    file1 = nn_filename[:-5] + "_part1" + ".nnet" #split the .nnet extenstion and add it at the end
    file2 = nn_filename[:-5] + "_part2" + ".nnet"
    writeNNet(weights1, biases1, mins1, maxes1, means1, ranges1, file1)
    writeNNet(weights2, biases2, mins2, maxes2, means2, ranges2, file2)
    return file1, file2


def ReLU(x):
    """
    Relu func for an np array
    :param x:
    :return:
    """
    return np.maximum(x, 0)

# def open_net(file):
#     return NNet(file)


def list_to_file(filename, lst):
    with open(filename, "w") as f:
        f.write("\n".join(lst))

def file_to_list(filename):
    with open(filename, "r") as f:
        lst = f.readlines()
    return lst

def validate_composition(orig_nn, nn1, nn2, inp):
    """
    Composition of the broken nn
    :param orig_nn:
    :param nn1:
    :param nn2:
    :param inp:
    :return:
    """
    orig_output = orig_nn.evaluate_network(inp)
    comp_output = nn2.evaluate_network(ReLU(nn1.evaluate_network(inp)))
    print(orig_output)
    print(comp_output)
    return np.all(orig_output == comp_output)


def build_equation(nn, line_arr):
    """
    Build equation from line
    :param line:
    :return:
    """
    eq = Equation(MarabouCore.Equation.LE if line_arr[-2] == "<=" else MarabouCore.Equation.GE)
    i=0
    curr = line_arr[i]
    while curr!="<=" and curr!=">=":
        sign = curr[0]
        y_index = curr.index('y')
        if y_index>1:
            coeff = curr[1:y_index]
            coeff = coeff if sign == '+' else -coeff
        else:
            coeff = 1 if sign == '+' else -1
        index = int(curr[y_index+1:])
        eq.addAddend(coeff, nn.outputVars[0][index])
        i += 1
        curr = line_arr[i]

    eq.setScalar(float(line_arr[-1]))
    return eq


def create_marabou_nn(nn_file, property_lines):
    """
    Create Marabou nnet from nnet file and property
    :param nn_file:
    :param property_lines:
    :return:
    """
    nn = Marabou.read_nnet(nn_file)
    for prop in property_lines:
        #set bound to input and output variables
        # split line s.t. 0=x_i, 1=leq/geq 2=value
        split_prop = prop.split(' ')
        if 'x' in prop:
            var_index = int(split_prop[0][1:])
            value = float(split_prop[2])
            if ">=" in prop:
                nn.setLowerBound(nn.inputVars[0][var_index], value)
            else:
                nn.setUpperBound(nn.inputVars[0][var_index], value)
        elif 'y' in prop:
            if len(split_prop) <= 3:
                var_index = int(split_prop[0][1:])
                value = float(split_prop[2])
                if ">=" in prop:
                    nn.setLowerBound(nn.outputVars[0][var_index], value)
                else:
                    nn.setUpperBound(nn.outputVars[0][var_index], value)
            else:
                eq = build_equation(nn, split_prop)
                nn.addEquation(eq)
    return nn

# create_marabou_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_1_3.nnet",
#                   file_to_list("/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt")).solve()

# def run():
#     orig_nn = NNet('ACASXU_experimental_v2a_2_7.nnet')
#     nn1 = NNet('ACASXU_experimental_v2a_2_7_part1.nnet')
#     nn2 = NNet('ACASXU_experimental_v2a_2_7_part2.nnet')
#     print(validate_composition(orig_nn, nn1, nn2, [-1,-1,-1,-1,-1]))
#     print(validate_composition(orig_nn, nn1, nn2, [15299.0,0.0,-3.1,600.0,500.0]))

# run()

"""
Run Marabou to check P=>R && R=>Q:
1. Build two files with properties for P=>R && R=>Q.
2. Build n files with P and one constraint from R.
3. Ran Marabou on p=>a_i, R=>Q (n+1 runnings).
4. If all running return UNSAT: return True, otherwise false.
"""

def verify_mid_property(nn_file1, nn_file2, prop_file, mid_prop_file):
    with open(prop_file) as f:
        k = 0 #k is the number of lines of the input property
        lines = list(f)
        n = len(lines)
        for l in lines:
            if 'y' in l:
                break
            k += 1
    os.system("touch second_prop.txt")
    os.system("cat " + mid_prop_file + " > second_prop.txt")
    os.system("cat " + prop_file + " | tail -" + str(n-k) + " >> second_prop.txt")

    os.system("touch output.txt")
    # os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou "+nn_file2+" "+" second_prop.txt | tail -1 > output.txt")
    os.system(
        "/cs/labs/guykatz/zivarda/Marabou/build/Marabou " + nn_file2 + " " + " second_prop.txt > output.txt")
    #check second property
    with open('output.txt') as f:
        result = f.readline().strip()
        if result != UNSAT:
            print("second")
            print("broken by second")
            # return NOT_MID
        else:
            print("second unsat")

    #check first property
    os.system("touch first_prop.txt")
    print("Verifiy first...")
    with open(mid_prop_file, 'r') as f:
        properties = f.read().splitlines()
        ind = 0
        for prop in properties:
            os.system("cat " + prop_file + " | head -" + str(k) + " > first_prop.txt")
            split_prop = prop.split(' ')
            not_prop = prop
            num = float(split_prop[2])
            if split_prop[1] == ">=":
                not_prop = not_prop.replace(">=", "\<=")
                num -= EPSILON
                not_prop = not_prop.replace(split_prop[2], str(num))
            else:
                not_prop = not_prop.replace("<=", "\>=")
                num += EPSILON
                not_prop = not_prop.replace(split_prop[2], str(num))
            # for s in split_prop:
            #     if s == ">=":
            #         not_prop += '\<= '
            #     elif s == '<=':
            #         not_prop += '\>= '
            #     else:
            #         not_prop += s + ' '
            not_prop = not_prop.replace('x', 'y')

            os.system('echo ' + not_prop + ' >> first_prop.txt')
            print("run marabou", ind)
            ind = ind+1
            os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou " + nn_file1 + " " + "first_prop.txt | tail -1 > output.txt")
            # print("finish marabou", flush=True)
            with open('output.txt') as fl:
                result = fl.readline().strip()
                if result != UNSAT:
                    # return NOT_MID
                    print("broken by first")

    return MID


# res = verify_mid_property("part1.nnet",
#                     "part2.nnet",
#                     "acas_property_3.txt",
#                     "mid_prop.txt")
#
# print(res)
#
# os.system("touch mid_prop.txt")
# f = open("mid_prop.txt", 'w')
# for i in range(50):
#     f.write("x"+str(i)+" >= 0\n")
# f.close()

# f = open("tree_property.txt", "w")
# vars = [41,21,2,40,43,9,33,22,26,27,34,3,8,42,4,15,20,23,37,45]
# # vars = [i-1 for i in orig_vars]
# state = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
# for i in range(len(vars)):
#     sign = " >= " if state[i]==1 else " <= "
#     f.write("x"+str(vars[i])+sign+str(0)+"\n")
#
# # for i in range(50):
# #     if i not in vars:
# #         f.write("x" + str(i) + " >= " + str(0) + "\n")
# f.close()



LAYER_SEPARATE = 5

def create_mid_property(nnet_file1, inp, new_filename):
    """
    Create a mid property by the first part of the net. The property created by the layer pattern of the given input
    at the end of the evaluation of nnet1
    :param nnet_file1: file of the part 1 of the original nnet
    :param inp: input to the net
    :param new_filename: name of the file with the mid property
    :return:
    """
    prop = ""
    # nn = NNet(nnet_file1)
    # output = nn.evaluate_network(inp)

    nn2 = Marabou.read_nnet(nnet_file1)
    output = nn2.evaluate([inp])[0]
    with open(new_filename, "w") as f:
        for i,j in enumerate(output):
            if j > 0:
                f.write("x"+str(i)+" >= 0\n")
                prop += "1"
            else:
                f.write("x"+str(i)+" <= 0\n")
                prop += "0"
    return prop

def run_all_nnets_and_properties(nnets_dir, properties_dir):
    print("START TSET")
    nnets_files = os.listdir(nnets_dir)
    properties_files = os.listdir(properties_dir)
    j = 0
    for nf in nnets_files:
        for pf in properties_files:
            # if(j<2):
            file1, file2 = split_nn(nnets_dir,nf, LAYER_SEPARATE)
            mid_prop_filename = "mid_prop" + nf[:-5] + pf[:-4] + ".txt"
            # inp = [1, 1, 1, 500, 600]
            # inp = [55947.691, 0, 0, 1150, 50]
            inp = [-0.3, 0, 0.5, 0.4, 0.4]
            create_mid_property(file1, inp, mid_prop_filename)
            res = verify_mid_property(file1, file2, properties_dir+"/"+pf, mid_prop_filename)
            print("Network:", nf, "Property:", pf, "Result:", res)
        # j += 1

# run_all_nnets_and_properties("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu",
#                              "/cs/labs/guykatz/zivarda/Marabou/resources/properties")

#*******counterexample*******
# print("start")
# print("split")
# f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", "ACASXU_experimental_v2a_1_1.nnet", 5)
# print("verify")
# res = verify_mid_property(f1, f2, "original_property.txt", "tree_property.txt")
# print(res)


#***I'm Here***
#after fix
# print("start")
# print("split")
# f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", "ACASXU_experimental_v2a_1_1.nnet", 5)
# print("verify")
# res = verify_mid_property(f1, f2, "paper_property.txt", "tree_property.txt")
# print(res)


# counterexample_input = [0.474418, 0.132776, -0.498410, 0.500000, 0.000000]
# net1 = NNet(f1)
# print(net1.evaluate_network(counterexample_input))
# nnMar = Marabou.read_nnet(f1)
# nnMar.setLowerBound(nnMar.inputVars[0][0], 0.268978)
# nnMar.setLowerBound(nnMar.inputVars[0][1], 0.111408)
# nnMar.setLowerBound(nnMar.inputVars[0][2], -0.5)
# nnMar.setLowerBound(nnMar.inputVars[0][3], 0.227273)
# nnMar.setLowerBound(nnMar.inputVars[0][4], 0.0)
#
# nnMar.setUpperBound(nnMar.inputVars[0][0], 0.679858)
# nnMar.setUpperBound(nnMar.inputVars[0][1], 0.499999)
# nnMar.setUpperBound(nnMar.inputVars[0][2], -0.49841)
# nnMar.setUpperBound(nnMar.inputVars[0][3], 0.5)
# nnMar.setUpperBound(nnMar.inputVars[0][4], 0.5)
#
# nnMar.setLowerBound(nnMar.outputVars[0][20], 0.1)
# options = Marabou.createOptions(verbosity=0)
# vals, stat = nnMar.solve(options=options)
# print(vals)

"""
In a loop:
1. evaluate some input
2. build layer pattern
2. verify each query with Marabou (50 queries)
3. if SAT - delete the corresponding constraint, 
"""
def build_layer_pattern(inp, property_file):
    f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", "ACASXU_experimental_v2a_1_3.nnet", 4)
    nnMar = Marabou.read_nnet(f1)

    #set bounds to input
    with open(property_file) as f:
        properties = f.read().splitlines()
        for prop in properties:
            if 'x' in prop:
                split_prop = prop.split(' ') #0=x_i, 1=leq/geq 2=value
                var_index = int(split_prop[0][1:])
                value = float(split_prop[2])
                if ">=" in prop:
                    nnMar.setLowerBound(nnMar.inputVars[0][var_index], value) 
                else:
                    nnMar.setUpperBound(nnMar.inputVars[0][var_index], value)



    #set bound for output
    nn = NNet(f1)
    output = nn.evaluate_network(inp)
    options = Marabou.createOptions(verbosity=0)

    # res = [0, 1, 3, 5, 6, 7, 8, 9, 12, 14, 15, 20, 21, 22, 24, 25, 26, 28, 31,
    #        33, 37, 38, 39, 42, 43, 45, 46, 48, 49]
    # vals = [(j>0) for i, j in enumerate(output) if i in res]

    create_mid_property(f1, inp, "mid_prop.txt")
    with open(property_file) as f:
        k = 0 #k is the number of lines of the input property
        lines = list(f)
        n = len(lines)
        for l in lines:
            if 'y' in l:
                break
            k += 1
    os.system("touch second_prop.txt")
    os.system("cat " + "mid_prop.txt" + " > second_prop.txt")
    os.system("cat " + property_file + " | tail -" + str(n-k) + " >> second_prop.txt")

    os.system("touch output.txt")
    # os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou "+f2+" "+" second_prop.txt")
    os.system(
        "/cs/labs/guykatz/zivarda/Marabou/build/Marabou " + f2 + " " + " second_prop.txt | tail -1 > output.txt")
    #check second property
    with open('output.txt') as f:
        result = f.readline().strip()
        if result != UNSAT:
            print("second")
            print("broken by second")
            # return NOT_MID
            return []
    print("second is unsat")

    to_remove = []
    vals_arr = []
    for i, j in enumerate(output): #i index, j val
        # if i<=40:
        #     continue
        print("Iteration", i)
        if j > 0: #the opposite of the comment
            nnMar.setUpperBound(nnMar.outputVars[0][i], 0)
        else:
            nnMar.setLowerBound(nnMar.outputVars[0][i], 0)
        vals, stat = nnMar.solve(options=options)

        if len(vals)!=0:
            to_remove.append(i)
            vals_arr.append(vals)

        if j > 0:
            del nnMar.upperBounds[nnMar.outputVars[0][i]]
        else:
            del nnMar.lowerBounds[nnMar.outputVars[0][i]]

    layer_pattern = [i for i in range(50) if i not in to_remove]
    # layer_pattern = [i for i in range(41,50) if i not in to_remove]


    print(layer_pattern)



    #run again second property
    os.system("touch second_prop_updated.txt")
    with open("mid_prop.txt") as f:
        with open("second_prop_updated.txt", "w") as g:
            lines = f.readlines()
            for (i, l) in enumerate(lines):
                if(i in layer_pattern):
                    g.write(l)

    os.system("cat " + property_file + " | tail -" + str(n - k) + " >> second_prop_updated.txt")
    os.system("touch output.txt")
    # os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou "+f2+" "+" second_prop.txt")
    os.system(
        "/cs/labs/guykatz/zivarda/Marabou/build/Marabou " + f2 + " " + " second_prop_updated.txt | tail -1 > output.txt")
    with open('output.txt') as f:
        result = f.readline().strip()
        if result != UNSAT:
            print("broken by second 2")
            # return NOT_MID
        else:
            print("unsat")

    # res = [0, 1, 3, 5, 6, 7, 8, 9, 12, 14, 15, 20, 21, 22, 24, 25, 26, 28, 31,
    #        33, 37, 38, 39, 42, 43, 45, 46, 48, 49]


    #[0, 1, 3, 5, 6, 7, 8, 9, 12, 14, 15, 20, 21, 22, 24, 25, 26, 28, 31, 33, 37, 38, 39, 42, 43, 45, 46, 48, 49]


    #if not work we try to split the property
    # mid_point = []
    # final_vals = vals_arr[0]
    # for i in range(nnMar.inputSize):
    #     mid_point.append(final_vals[nnMar.inputVars[0][i]])
    # split_property(property_file, "input_property_split", mid_point, 0.000001)


    # for i, j in enumerate(output):
    #     if j > 0:
    #         nnMar.setLowerBound(nnMar.outputVars[0][i], 0)
    #     else:
    #         nnMar.setUpperBound(nnMar.outputVars[0][i], 0)
    
    # arr = [i for i in range(50)]
    # while len(arr)!=0:
    #     vals, stat = nnMar.solve()
    #     if len(vals) == 0:
    #         print("UNSAT")
    #         break

        #there is contradiction
        # for j in range(len(nnMar.inputVars)):
        #     v = vals[nnMar.inputVars[j]]


def split_property(property_file, output_filename, orig_point, break_point):
    """
    Split property input region
    :param property_file:
    :param output_filename:
    :param orig_point:
    :param break_point:
    :return:
    """
    with open(property_file, "r+") as pf:
        #f1 = file with orig_point
        with open(output_filename+"1.txt", 'w') as opf1:
            #f2 - file with break_point
            with open(output_filename+"2.txt", 'w') as opf2:
                lines = pf.readlines()
                for line in lines:
                    if "x" in line:
                        parts = line.split(' ')
                        index = int(parts[0][1:]) #remove 'x' char
                        value1 = orig_point[index]
                        value2 = break_point[index]
                        avg = (value1+value2)/2
                        if parts[1] == ">=":
                            if value1 < value2:
                                opf1.write(line)
                                opf2.write(parts[0] + " " + parts[1] + " " + str(avg)+"\n")
                            else:
                                opf1.write(parts[0] + " " + parts[1] + " " + str(avg)+"\n")
                                opf2.write(line)


                        if parts[1] == "<=":
                            if value1 < value2:
                                opf1.write(parts[0] + " " + parts[1] + " " + str(avg) + "\n")
                                opf2.write(line)
                            else:
                                opf1.write(line)
                                opf2.write(parts[0] + " " + parts[1] + " " + str(avg) + "\n")


                    else:
                        opf1.write(line)
                        opf2.write(line)

def split_property2(pf_lines, orig_point, break_point):
    """
    Split property input region
    :param property_file:
    :param output_filename:
    :param orig_point:
    :param break_point:
    :return: new properties in an array
    """
    op1 = []
    op2 = []
    for line in pf_lines:
        if "x" in line:
            parts = line.split(' ')
            index = int(parts[0][1:]) #remove 'x' char
            value1 = orig_point[index]
            value2 = break_point[index]
            avg = (value1+value2)/2
            if parts[1] == ">=":
                if value1 < value2:
                    op1.append(line)
                    op2.append(parts[0] + " " + parts[1] + " " + str(avg)+"\n")
                else:
                    op1.append(parts[0] + " " + parts[1] + " " + str(avg)+"\n")
                    op2.append(line)

            if parts[1] == "<=":
                if value1 < value2:
                    op1.append(parts[0] + " " + parts[1] + " " + str(avg) + "\n")
                    op2.append(line)
                else:
                    op1.append(line)
                    op2.append(parts[0] + " " + parts[1] + " " + str(avg) + "\n")
        else:
            op1.append(line)
            op2.append(line)
    return op1, op2


def split_property3(pf_lines, orig_point, break_point, input_pattern):
    """
    Split property input region and get a point in the region
    :param property_file:
    :param output_filename:
    :param orig_point:
    :param break_point:
    :param input_pattern: pattern of 0/1 which definde in which part in the space we are
    :return: new properties in an array
    """
    op = []
    point_for_pattern = [0]*5
    for line in pf_lines:
        if "x" in line:
            parts = line.split(' ')
            index = int(parts[0][1:]) #remove 'x' char
            value1 = orig_point[index]
            value2 = break_point[index]
            avg = (value1+value2)/2
            if parts[1] == ">=":
                if input_pattern[index] == '0':
                    op.append(line)
                    point_for_pattern[index] = (avg+float(parts[2]))/2
                else:
                    op.append(parts[0] + " " + parts[1] + " " + str(avg)+"\n")

            if parts[1] == "<=":
                if input_pattern[index] == '0':
                    op.append(parts[0] + " " + parts[1] + " " + str(avg) + "\n")
                else:
                    op.append(line)
                    point_for_pattern[index] = (avg + float(parts[2])) / 2
        else:
            op.append(line)
            op.append(line)
    if input_pattern == '0'*INPUT_SIZE:
        point_to_return = orig_point
    elif input_pattern == '1'*INPUT_SIZE:
        point_to_return = break_point
    else:
        point_to_return = point_for_pattern

    return op, point_to_return

def verify_first_part(f1, property_file_lines, point):
    nnMar = Marabou.read_nnet(f1)

    # set bounds to input
    for prop in property_file_lines:
        if 'x' in prop:
            split_prop = prop.split(' ')  # 0=x_i, 1=leq/geq 2=value
            var_index = int(split_prop[0][1:])
            value = float(split_prop[2])
            if ">=" in prop:
                nnMar.setLowerBound(nnMar.inputVars[0][var_index], value)
            else:
                nnMar.setUpperBound(nnMar.inputVars[0][var_index], value)

    # set bound for output

    # nn = NNet(f1)
    # output = nn.evaluate_network(point)
    output = (nnMar.evaluate([point]))[0]

    options = Marabou.createOptions(verbosity=0)

    to_remove = []
    vals_arr = []
    for i, j in enumerate(output):  # i index, j val
        # if i<=40:
        #     continue
        # print("Iteration", i)
        if j >= 0:  # the opposite of the comment
            nnMar.setUpperBound(nnMar.outputVars[0][i], -0.0001)
        else:
            nnMar.setLowerBound(nnMar.outputVars[0][i], 0.0001)
        vals, stat = nnMar.solve(options=options, verbose=False)

        if len(vals) != 0: #sat
            to_remove.append(i)
            vals_arr.append(vals)

            break_point = []
            for k in range(nnMar.inputSize):
                break_point.append(vals[nnMar.inputVars[0][k]])
            return {"status": NOT_MID, "point": break_point, "index": i}

        if j > 0:
            del nnMar.upperBounds[nnMar.outputVars[0][i]]
        else:
            del nnMar.lowerBounds[nnMar.outputVars[0][i]]

    layer_pattern = [i for i in range(50) if i not in to_remove]

    # print(layer_pattern)
    return {"status": MID, "point": None, "index": None}


def layer_pattern_by_split_input_region(f1, f2, property_file_lines, point, depth):
    print("start", depth)
    #first, verify that the second is unsat
    mid_prop = create_mid_property(f1, point, "mid_prop.txt")
    print(mid_prop)
    k = 0  # k is the number of lines of the input property
    # lines = list(f)
    # n = len(lines)
    for l in property_file_lines:
        if 'y' in l:
            break
        k += 1
    os.system("touch second_prop.txt")
    os.system("cat " + "mid_prop.txt" + " > second_prop.txt")
    # os.system("cat " + property_file + " | tail -" + str(n - k) + " >> second_prop.txt")
    with open("second_prop.txt", "a") as f:
        f.write("".join(property_file_lines[k:]))

    os.system("touch output.txt")
    # os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou "+f2+" "+" second_prop.txt")
    os.system(
        "/cs/labs/guykatz/zivarda/Marabou/build/Marabou " + f2 + " " + " second_prop.txt | tail -1 > output.txt")
    # # check second property
    with open('output.txt') as f:
        result = f.readline().strip()
        if result != UNSAT:
            print("second")
            print("broken by second")
            # return NOT_MID
            # return
    print("second is unsat")
    res = verify_first_part(f1, property_file_lines, point)
    if res["status"] == NOT_MID:
        if len(depth) == 15:
            print("stop recursion in depth", depth)
            return
        # op1, op2 = split_property2(property_file_lines, point, res["point"])
        # layer_pattern_by_split_input_region(f1, f2, op1, point, depth+"1")
        # layer_pattern_by_split_input_region(f1, f2, op2, res["point"], depth+"2")
        for i in range(2**INPUT_SIZE):
            rep = bin(i)[2:].zfill(5)
            op, point_for_pattern = split_property3(property_file_lines, point, res["point"], rep)
            layer_pattern_by_split_input_region(f1, f2, op, point_for_pattern,depth+str(i))
        print("finish", depth)

    else:
        print(MID, "in depth", depth)
        with open("layer_pattern_output.txt", "a") as f:
            f.write(depth+": "+mid_prop+"\n")
    # i=0
    # res = verify_first_part(f1, property_file_lines, point)
    # op1 = property_file_lines
    # while res["status"] == NOT_MID:
    #     print("Split:", i, "Neuron:", res["index"])
    #     op1, op2 = split_property2(op1, point, res["point"])
    #     res = verify_first_part(f1, op1, point)
    #     i += 1
    #
    # print("Total partition number:", i)

    # res = verify_first_part(f1, property_file_lines, point)
    # curr_property_file = property_file
    # i = 0
    # while res["status"] == NOT_MID:
    #     print("Split:", 0, "Neuron:", res["index"])
    #     split_property(curr_property_file, "input_property_split", point, res["point"])
    #
    #
    #     shutil.copy2("input_property_split1.txt", "input_property_split_curr.txt")
    #     curr_property_file = "input_property_split_curr.txt"
    #     res = verify_first_part(f1, file_to_list(curr_property_file), point)
    #     i += 1
    #
    # print("Total partition number:", i)
    # print("Verify...")
    # print(verify_mid_property(f1, f2, curr_property_file, "mid_prop.txt"))

"""
Algorithm:
*choose a point and build layer pattern
*try to verify
*if sat: 
-we have a point that not satisfies the pattern
-split the input space with mid point between original point and our point
-recursively check two regions
"""


def verify_second_part(nn_file, property_file_lines, point):
    LAYER = 4
    f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", nn_file, LAYER)

    # check if the second is unsat
    mid_prop = create_mid_property(f1, point, "mid_prop.txt")
    print(mid_prop)
    k = 0  # k is the number of lines of the input property
    for l in property_file_lines:
        if 'y' in l:
            break
        k += 1
    os.system("touch second_prop.txt")
    os.system("cat " + "mid_prop.txt" + " > second_prop.txt")
    # os.system("cat " + property_file + " | tail -" + str(n - k) + " >> second_prop.txt")
    with open("second_prop.txt", "a") as f:
        f.write("".join(property_file_lines[k:]))

    os.system("touch output.txt")
    # os.system("/cs/labs/guykatz/zivarda/Marabou/build/Marabou "+f2+" "+" second_prop.txt")
    nn_second = create_marabou_nn(f2, file_to_list("second_prop.txt"))
    vals, stat = nn_second.solve()
    curr_prop = property_file_lines
    itr=0
    while len(vals)!=0:
        new_prop = []
        for p in curr_prop:
            if 'x' in p:
                split_prop = p.split(' ')  # 0=x_i, 1=leq/geq 2=value
                var_index = int(split_prop[0][1:])
                value = float(split_prop[2])
                avg = (value+point[var_index])/2
                new_prop.append(split_prop[0]+" "+split_prop[1]+" "+str(avg))
            else:
                new_prop.append(p)

        nn = create_marabou_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu/"+nn_file, new_prop)
        p = multiprocessing.Process(target=nn.solve)
        p.start()
        p.join(3)
        if p.is_alive():
            p.kill()
            p.join()
        # nn.solve()
        change = False
        with open("lowerBounds.txt", "r") as lb_f:
            with open("upperBounds.txt", "r") as ub_f:
                lower_bounds = lb_f.readlines()[LAYER].split(',')
                upper_bounds = ub_f.readlines()[LAYER].split(',')
                for i in range(len(lower_bounds)-1):
                    var_index = nn_second.inputVars[0][i]
                    if nn_second.getLowerBound(0, var_index) < float(lower_bounds[i]) < nn_second.getUpperBound(0, var_index):
                        nn_second.setLowerBound(var_index, float(lower_bounds[i]))
                        change = True
                    if nn_second.getUpperBound(0, var_index) > float(upper_bounds[i]) > nn_second.getLowerBound(0, var_index):
                        nn_second.setUpperBound(var_index, float(upper_bounds[i]))
                        change = True
        if change:
            vals, stat = nn_second.solve()
        curr_prop = new_prop
        itr += 1
    print("second is UNSAT", "Iterations:", itr)
    print(curr_prop)

#***I'm here2
#[-0.301041984, 0, 0, 0.4090909091, 0.125]
#[-0.3, 0.0095492966, 0, 0.5, 0.1666666667]

# with open("layer_pattern_output.txt", "w"):
#     pass
# f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", "ACASXU_experimental_v2a_1_3.nnet", 4)
# layer_pattern_by_split_input_region(f1, f2,
# file_to_list("/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt"),
#                                     [-0.3, 0.0095492966, 0, 0.5, 0.1666666667], "1")

verify_second_part("ACASXU_experimental_v2a_1_3.nnet",
                   file_to_list("/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt"),
                   [-0.3, 0.0095492966, 0, 0.5, 0.1666666667])

# layer_pattern_by_split_input_region(f1, f2,
# file_to_list("/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt"),
#                                     [-0.301, 0, 0, 0.4, 0.1], "1")
        
# build_layer_pattern([-0.3, 0.0095492966, 0, 0.5, 0.1666666667], "/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt")


# [-0.3035311561, -0.0095492966, 0.0, 0.45585841897798146, 0.0833333333]

# split_property("/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt",
#                "input_property_split",
#                [-0.30131161232628967, 0.0033015103745909985, 0.0, 0.4722566939034796, 0.12235222380125145]
#                , 0.00000001)

# build_layer_pattern([-0.3, 0.0095492966, 0, 0.5, 0.1666666667], "input_property_split1.txt")


# build_layer_pattern([-0.3, 0.0095492966, 0, 0.5, 0.1666666667], "input_property_split2.txt")
#***




# f1, f2 = split_nn("/cs/labs/guykatz/zivarda/Marabou/resources/nnet/acasxu", "ACASXU_experimental_v2a_1_3.nnet", 4)
# verify_mid_property(f1, f2, "/cs/labs/guykatz/zivarda/Marabou/resources/properties/acas_property_4.txt", "inputMid.txt")


# [-0.3, 0.0095492966, 0, 0.5, 0.1666666667]
# [0, 1, 3, 5, 6, 7, 8, 9, 12, 14, 15, 20, 21, 22, 24, 25, 26, 28, 31, 33, 37, 38, 39, 42, 43, 45, 46, 49]


#layer5 not work
#unsat_iter = [1,5,7,8,9,14,15,16,17,18,19,22,23,24,26,28,30,31,32,33,36,37,39] (till 40)
#[1, 5, 7, 8, 9, 14, 15, 16, 17, 18, 19, 22, 23, 24, 26, 28, 30, 31, 32, 33, 36, 37, 39, 41, 42, 44, 45, 48, 49]


#[1,5,7,8,9,14,15,16,17,18,19,22,23,24,26,28,30,31,32,33,36,37,39,]



#todo: *property & nnet file to Maraboupy nnet --V
#   *update bound by gurobi output file
#   *add parameter that stops running after gurobi
#   *update main function