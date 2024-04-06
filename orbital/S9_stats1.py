# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt


# This script is decpreciated


# %% Import Data

# Data from S9_distribution1, no monte-carlo
col_ang_list = [0.9254841042983534, 0.9451847049179114, 0.9361830040913306, 0.932394335915644, 0.9313336029002588,
                0.9191634392393755, 0.9203258976236081, 0.9186508112507988, 0.9084310837360605, 0.9131255151077752,
                0.9178219159426579, 0.9233892094586944, 0.9308215859332335, 0.9356080742328672, 0.9508997400341044,
                0.9560195630874098, 0.9554714819169532, 0.9535330805408221]

swe_ang_list = [-0.2378429951983393, -0.5459437179067221, -0.24431717045850176, -0.5191887985743285,
                -0.24755478264552075, -0.5323931942190155, -0.28994425939034935, -0.5972717949147741,
                -0.31808423290844523, -0.6427440571227279, -0.252432603715733, -0.5380246401728652,
                -0.17858699716000637, -0.4941362666397597, -0.11775716180509893, -0.40782694852048806,
                -0.23371890654964306, -0.5274509243637837]

integral_list = [1926.6309711424854, 1927.32883045908, 1930.6309537681377, 1924.8944944068148, 1925.639592246792,
                 1923.0574482643815, 1926.8534947190765, 1923.7759562082103, 1925.9689007428558, 1924.7157281003617,
                 1926.0848119239859, 1925.2157190061657, 1927.241601877067, 1925.888769852068, 1926.3508614247387,
                 1922.8364491839795, 1923.5283114665137, 1926.4830357329138]

plot_0 = [1926.6309711424854, 1926.096920773096, 1926.5347297100477, 1926.3778471181704, 1926.4102589249196, 1926.5778471742215, 1926.0599688043292, 1926.8638035112042, 1926.2172346454356, 1926.332773429305, 1926.8258922009206, 1926.2026245864276, 1926.1963107168156, 1926.5798690032623, 1926.7851778832296, 1926.8909135610575, 1926.0228321726631, 1926.3378139695456, 1926.1307442460009, 1926.621566217788, 1925.9772964730882, 1926.470884625003, 1926.6449200716822, 1926.5651379059536, 1926.4442604581936, 1926.4449904211704, 1926.5537262706387, 1926.3551608115213, 1925.9890260296318, 1926.6698093952025, 1929.221728165428, 1925.8336149708905, 1925.9057356540184, 1925.5214953794814, 1925.917444937631, 1926.6185370919943, 1926.4988422191734, 1926.511773743361, 1926.9368702777156, 1925.6966534823387, 1926.3870597607092, 1927.0122288197522, 1925.758613495733, 1925.9036571047131, 1926.759086698071, 1926.9377077160605, 1926.22719365785, 1926.878540698804, 1926.3622987220147, 1926.4491377840072]
plot_1 = [1927.32883045908, 1927.8339599122994, 1926.9794616142137, 1927.5766440089808, 1927.2199996295537, 1927.4168930267133, 1927.4082152220105, 1926.9215440104567, 1927.3597614397468, 1927.380260269522, 1927.9335232650653, 1927.4856510651346, 1927.3853272884558, 1927.64137808135, 1927.1304474987962, 1927.8042047341269, 1927.2022538725037, 1927.4766031084407, 1927.0916960656034, 1927.408104935105, 1927.35574634684, 1927.8360302745723, 1927.305017681825, 1927.622438026937, 1926.9234558968938, 1927.2590101572555, 1927.5039763870805, 1927.1340577490898, 1927.179039112893, 1927.2076983012307, 1927.3600444750641, 1927.214260402264, 1927.0287757553137, 1927.2341172496845, 1927.370578932052, 1927.4700688037533, 1927.3931731374162, 1927.348540469435, 1927.436540678849, 1927.1535479827785, 1926.9587295626668, 1927.506453906714, 1927.9966674499772, 1927.4266587383327, 1927.355603237898, 1927.4202620827189, 1927.7085170271741, 1927.1427145179825, 1927.0688722022828, 1927.717617507075]
plot_2 = [1930.6309537681377, 1933.4613740056602, 1930.5849559080832, 1930.9486306292945, 1930.8130143926833, 1930.1853931849503, 1930.6541422396892, 1931.1480076996224, 1933.7550768675019, 1931.0505629879874, 1930.2310478310917, 1933.3238979654325, 1933.7227299324747, 1930.2773685228995, 1933.7476456954817, 1933.588429006209, 1933.2658590874944, 1930.2081688890355, 1931.0866406506427, 1930.5222700091776, 1930.3516478843294, 1930.6794320048768, 1930.2839347662448, 1930.6880583148995, 1930.3813231575305, 1930.6050783905325, 1930.864157868038, 1933.9849704890635, 1930.3035468187463, 1930.9337683002748, 1930.0377328462332, 1933.9662304209444, 1933.923188211769, 1933.1138970453087, 1933.2825907826639, 1930.4162421489384, 1931.038047952718, 1930.672443974321, 1933.044257418067, 1930.6479117636413, 1930.810941478753, 1930.3726036865332, 1930.3213570920298, 1930.6196273384735, 1930.7557198761065, 1930.73257191111, 1930.4578183087715, 1930.9008074197025, 1933.5204283652072, 1930.8116474314743]
plot_3 = [1924.8944944068148, 1925.0096040199737, 1924.818181748949, 1924.95553917358, 1924.918884971257, 1925.001796527963, 1924.7932369074833, 1925.0745531530706, 1924.937920699675, 1924.7016841902944, 1925.068819113764, 1924.9579635091663, 1925.2269112622268, 1925.096172347274, 1924.65361147238, 1925.040099960354, 1924.896763541949, 1925.3266308845355, 1924.494847923124, 1924.3980704437556, 1924.731997548154, 1925.2700756998881, 1925.0203618565101, 1925.2649069612835, 1924.7275836389558, 1925.4469623392, 1925.0000974815061, 1924.9575691268012, 1925.005256867834, 1925.1965053861795, 1924.9332399575449, 1924.9786980126144, 1925.1653379966033, 1924.8835641330206, 1924.2934214042793, 1924.5371086154555, 1924.9005796919596, 1924.8821028536413, 1924.8921512608476, 1925.0220622043921, 1924.5353682738248, 1924.9524649059792, 1925.0069282763318, 1925.2944765502637, 1925.1357534888305, 1924.3816935994146, 1924.6491750979674, 1925.1129227941817, 1925.0738953658745, 1924.8779538410379]
plot_4 = [1925.639592246792, 1925.6334899328367, 1925.9437089381358, 1925.9473181955398, 1925.3856843458493, 1926.2336184839703, 1925.846248322152, 1925.2280749466258, 1925.5391520213022, 1925.70137035054, 1928.2995326186947, 1925.5040116268679, 1925.1809392609136, 1925.3904327327662, 1926.0816099233398, 1928.9901426863887, 1925.9306231508235, 1925.1684111372958, 1925.701817701801, 1929.2787038701965, 1925.168210134807, 1925.9717707515902, 1925.2552262403428, 1925.7759584268924, 1925.4940555239166, 1929.0524223802445, 1928.404909870334, 1925.5921233152146, 1925.5796650216153, 1925.688081990569, 1925.0878906614525, 1925.462415799483, 1925.6005798519147, 1929.088005092511, 1929.1350839334816, 1925.2487891763706, 1925.5634656997943, 1925.8173602207062, 1925.2051295732822, 1925.649978681101, 1925.5377041222594, 1925.4485143903626, 1926.0917578792378, 1925.9575257791016, 1926.032119534255, 1925.2363579474859, 1925.572583413084, 1926.0642837768141, 1925.7944077841682, 1925.6763305436618]
plot_5 = [1923.0574482643815, 1922.729248648892, 1922.5996145561344, 1923.2576427848262, 1922.952056865206, 1923.0131593067401, 1923.3235256050284, 1923.217257387594, 1923.492878114703, 1923.3368131640632, 1923.1425182460794, 1922.7427877883783, 1923.415030092772, 1923.2149255344282, 1923.0939742887513, 1923.0918578656233, 1922.9107526587275, 1923.3138985784228, 1923.0126821497597, 1922.560139123343, 1923.0898276306084, 1923.2749338640358, 1923.4282857690066, 1923.2185220682873, 1923.70610302611, 1923.3714438465838, 1922.9198780929148, 1923.0737378513354, 1923.4794307081938, 1922.9867825958047, 1922.7866399406496, 1923.3638625940282, 1923.0675938333318, 1923.1396490000382, 1922.939740334682, 1923.274579398697, 1922.8671770277542, 1923.016128214119, 1922.7610356977455, 1923.1006652702713, 1923.2668325166212, 1922.900979355177, 1922.7221788881227, 1922.901253660585, 1922.9461457550856, 1923.0066455925007, 1923.1351713631177, 1922.6177897442706, 1922.8622419529715, 1923.1391947155437]
plot_6 = [1926.8534947190765, 1927.2847780217958, 1926.514012132363, 1926.7688822694656, 1926.8712840655196, 1927.1443609276373, 1926.4155154069833, 1926.8341830457032, 1927.1474257740172, 1927.0341219584984, 1926.8176573014666, 1926.5052509767872, 1926.970107739815, 1926.4034654039328, 1926.6273899532923, 1926.9967629087205, 1926.523538030767, 1927.0619451427779, 1926.8649660907522, 1927.184861373847, 1926.909873497878, 1927.0862258143886, 1927.3365817257481, 1926.9354837362357, 1926.4860367246663, 1927.2690555414267, 1926.4509276781557, 1926.9347135652386, 1927.2176773319454, 1926.7407256286426, 1926.1064249026276, 1927.1445814300964, 1926.6178871203049, 1926.5131579195445, 1926.799141937878, 1927.1187961861438, 1926.1160340920044, 1926.9573624436275, 1926.5326186307902, 1926.8947472064174, 1926.9049894470377, 1926.8336294625315, 1927.0488346154586, 1926.3894764241072, 1927.074144298336, 1926.9230537498, 1926.965388418417, 1927.007832236077, 1926.8088994676225, 1927.001475389308]
plot_7 = [1923.7759562082103, 1923.877729656839, 1924.2630463434114, 1923.7655328420785, 1923.537085693635, 1923.541351449285, 1923.6907085018443, 1923.4666677774176, 1923.7896813123298, 1923.7195245443972, 1923.271797141977, 1923.06591242114, 1923.4293178006749, 1923.7400038157245, 1923.8057718240132, 1923.5341777311617, 1923.3179807440438, 1923.943443380071, 1923.2839360593553, 1923.4543419519189, 1923.2748486350358, 1923.8464754698866, 1923.867558925715, 1924.0959973215638, 1923.7111818028304, 1924.277963675344, 1923.5012072684021, 1923.8076051180317, 1923.4283787990325, 1923.755385562415, 1923.5623494204215, 1923.6815806125462, 1923.8166509108169, 1923.2073906435544, 1923.8741734938083, 1924.0406955700857, 1923.9725650998607, 1923.7696515807438, 1923.6743609575444, 1923.7313638700914, 1924.1410666834158, 1923.4986148481792, 1923.583771887934, 1923.7602464951026, 1923.279389827677, 1923.3167207252764, 1923.4142999787207, 1923.5471975230585, 1924.1920898896024, 1923.908456539306]
plot_8 = [1925.9689007428558, 1926.290816371229, 1925.9003965035772, 1926.041069971919, 1925.8370421436346, 1925.8762490652196, 1926.2761503113832, 1926.1656048487039, 1926.0180162350125, 1925.5590923367881, 1925.839781655563, 1926.0090608969028, 1926.7278616368621, 1926.0405320622008, 1925.8930508135759, 1926.5893295307048, 1925.862604853323, 1925.5391759549598, 1926.0905120363348, 1926.2244037676433, 1925.8203698267223, 1926.1035696913334, 1925.5185121885484, 1925.6817409723146, 1925.9411977511384, 1925.838783687989, 1925.7124561783116, 1926.2648972322932, 1925.7881110351625, 1926.2127158243077, 1925.7635046275004, 1925.8760499883908, 1925.8716648427558, 1926.0013406597097, 1926.0351670230384, 1925.9406173361044, 1925.7479168640325, 1925.9384519487448, 1925.9522116931487, 1925.4547654451796, 1926.2130932682228, 1925.7233064229151, 1926.2136152869596, 1925.9879860763488, 1925.948882883521, 1926.085076726226, 1926.0302631869417, 1925.7654782640225, 1925.581406567231, 1925.8673793989146]
plot_9 = [1924.7157281003617, 1924.5682522313334, 1924.7439775401767, 1925.1002858390814, 1924.6647167604062, 1924.5523152294995, 1925.154769354886, 1924.476441682719, 1924.9145326387736, 1924.638176033118, 1924.5478622416085, 1924.7123837197423, 1925.0046834679542, 1924.9577137452943, 1925.0498004715885, 1924.8736782757612, 1924.7364618655056, 1924.5740572249538, 1924.900042772509, 1924.4545901133463, 1924.491115829143, 1924.761269465906, 1924.4993390061002, 1925.0699022352724, 1924.3455478527842, 1924.4753437530107, 1924.1933062518194, 1924.8913068882428, 1924.8112402271047, 1924.7576039148494, 1924.6170600617208, 1924.9434360603416, 1924.8990216016205, 1924.6307457144644, 1924.8846695338912, 1924.6317379578663, 1924.7436689672493, 1924.4836383402355, 1924.8461077481693, 1924.6337596509863, 1924.8474583496684, 1924.8301562451002, 1924.6470723913565, 1924.8999623953455, 1924.3125569453355, 1924.5905358584498, 1924.793970602929, 1924.4534404584988, 1924.271591025873, 1924.6558508766072]
plot_10 = [1926.0848119239859, 1925.6978634051488, 1926.027112623817, 1926.0588298479931, 1925.9033678534545, 1926.0279656359028, 1926.1954733608104, 1926.1166879430414, 1925.8632199973983, 1926.1431945866898, 1925.7314357877565, 1926.4482827455647, 1926.0024182764632, 1925.8526086082682, 1926.0744164061718, 1925.8418194165954, 1925.6412257890192, 1926.3002225732457, 1926.119516562168, 1926.720277263433, 1925.9230665155706, 1925.682459836308, 1926.098693291952, 1926.1147021830584, 1926.3840987890464, 1926.4116072591714, 1925.9916943920603, 1926.613385789582, 1925.8544019201865, 1926.141499919187, 1925.9561113663497, 1926.9669716701378, 1926.3841959127333, 1925.8854252599024, 1926.4670076737991, 1926.4556870665729, 1925.7470112686037, 1926.77624407423, 1926.7385616088325, 1925.824176204581, 1926.2464277714298, 1925.7891317580957, 1926.0333798638294, 1926.4379623701134, 1925.8553683353111, 1926.49370553831, 1926.3509898376058, 1925.897576715005, 1925.858883361625, 1926.1899473772462]
plot_11 = [1925.2157190061657, 1925.123641287785, 1925.2208458558625, 1925.0386273228316, 1925.4898420683073, 1925.3450759749442, 1925.1262110198104, 1925.0588642091539, 1925.343625731322, 1925.3501523047491, 1925.3981191888238, 1924.9766123989198, 1924.827608572738, 1925.395596493488, 1925.0059426725468, 1925.471139922678, 1925.2851195486435, 1924.8135528427888, 1925.2752312143598, 1925.2503628020054, 1925.1836047129236, 1924.9593100002437, 1925.2412753956166, 1925.2262649318557, 1925.5967479721005, 1925.228312089438, 1925.0671367957607, 1925.1832127547964, 1925.6298449724718, 1925.268670598568, 1925.1144525053835, 1925.1856884961082, 1925.1695874811628, 1925.4716389369214, 1925.3589133003334, 1924.9603015390912, 1925.4346936988195, 1925.3028556912777, 1925.0340789946492, 1925.212462141546, 1925.3396506929898, 1925.485438655414, 1925.1345710460296, 1925.640345700541, 1925.0211876162928, 1925.3786795303029, 1925.165789159473, 1925.069711027984, 1925.2102383343824, 1924.769626975229]
plot_12 = [1927.241601877067, 1926.6162783666382, 1927.0541961910026, 1927.3217280453957, 1927.823354151875, 1927.330285262007, 1925.5196122364753, 1926.7932257264863, 1926.740986261949, 1927.0127581200597, 1927.0664386065662, 1926.7461979476375, 1927.038496595049, 1927.219844518045, 1927.1935579997923, 1927.170730059605, 1927.2896854483777, 1927.8097748508007, 1927.3879546023118, 1927.0253327850564, 1927.4285357469746, 1927.5110432263477, 1926.8887567744905, 1927.4598464253297, 1927.379625388631, 1925.7287592579162, 1926.886855174902, 1926.6073217112703, 1927.0758552239295, 1926.9593106030115, 1927.5174474883288, 1927.0788084577946, 1927.5723612770985, 1927.0081165366657, 1927.3974647485152, 1927.092638571058, 1927.1144733472488, 1926.9863190910594, 1927.0356536061427, 1927.169372504449, 1927.6554274951702, 1927.1082003155518, 1927.6577877079399, 1926.8753625166662, 1926.9556097416414, 1927.0841006521146, 1927.3694781519857, 1927.3585129104677, 1926.8612583711815, 1927.0749761292166]
plot_13 = [1925.888769852068, 1925.850916240846, 1926.1854797956087, 1925.9292092408732, 1925.3399381583204, 1925.5035497321626, 1926.3623876344375, 1925.7308377110235, 1925.9119287507647, 1926.0518909676327, 1926.272351890881, 1926.0152753731154, 1926.0818185460055, 1925.3777839471134, 1925.9563737371607, 1925.7596056839195, 1925.7563650891464, 1925.5368474447157, 1925.95808367193, 1925.8053209990505, 1925.793285186834, 1925.8228300079202, 1925.7976252962765, 1925.9999753014365, 1925.7640303548746, 1926.1526714757201, 1925.6116324693448, 1925.7372881879924, 1925.9088365498376, 1926.3435396997995, 1925.949226677773, 1925.6233314516396, 1925.6641216282399, 1926.1014640752728, 1925.698061629708, 1925.821570727892, 1926.1604359272985, 1925.7172641512561, 1925.5152757687374, 1925.797037689964, 1926.2506931561982, 1925.7156771211885, 1925.6160851977613, 1925.5912160095727, 1925.7951588836056, 1926.3470064686676, 1925.9030172817704, 1926.1317603066423, 1925.5899974491049, 1926.0555625941108]
plot_14 = [1926.3508614247387, 1926.4644538469356, 1926.2115204599952, 1926.552773708365, 1926.4605871217937, 1926.3529306222936, 1926.4704900541503, 1926.282641156958, 1926.4409437802517, 1926.3981830345458, 1926.6558386981362, 1926.2986696699697, 1926.0856433503368, 1925.8067391524726, 1926.3567601226432, 1926.3335068426115, 1926.1425661463386, 1926.2904749587615, 1926.0202522913667, 1926.7433498735345, 1926.3970141602376, 1926.802456025747, 1925.8410238749711, 1926.361841889341, 1926.3032906248661, 1926.2435004929607, 1926.565182349517, 1926.294322929988, 1926.39146821129, 1926.0975479420572, 1926.4410081887497, 1926.557084047381, 1926.0267735056752, 1926.1243485256894, 1926.5599958323394, 1926.461385847029, 1926.4909905985078, 1926.616309387989, 1926.3077580738668, 1926.5561555434288, 1926.1038965412772, 1926.6536133333939, 1926.1194818609888, 1926.439919861659, 1926.3449422375995, 1925.9278638120986, 1926.3639765625264, 1926.5363517611468, 1926.1182047519953, 1926.2860740334731]
plot_15 = [1922.8364491839795, 1922.600750935798, 1922.3313803191672, 1923.0207526312117, 1922.8218222852245, 1922.9664593262262, 1922.643707796903, 1922.3990372035457, 1922.711942588966, 1922.7064295482035, 1922.675069827129, 1922.5462987812493, 1922.936030019379, 1922.3579846516457, 1923.0376122429338, 1922.638121310518, 1922.809151004257, 1922.8431870346703, 1922.3456849180757, 1922.7287508005636, 1923.1678887592766, 1922.6926341878604, 1922.7738375576341, 1923.3157599111216, 1922.6924759666313, 1922.9811734725477, 1922.7578371036163, 1922.839109802617, 1922.5731482613223, 1922.4054961405454, 1922.5116976423205, 1922.4799589994675, 1922.597073285381, 1923.5065650376146, 1923.1338662129867, 1923.0691299437538, 1922.9543110655918, 1922.3233600287267, 1922.5817049949703, 1922.8610075291735, 1923.094118217126, 1922.4212823142807, 1922.7782341398192, 1922.943957406447, 1922.5217465769624, 1922.8724759112931, 1922.9175500246047, 1923.3558726334986, 1922.2673414094966, 1923.0032598432635]
plot_16 = [1923.5283114665137, 1923.6939308389356, 1923.2491160435775, 1923.794111793782, 1923.3563639413399, 1923.1764545560618, 1923.6871477118596, 1923.8384036732646, 1923.4403498587506, 1923.801200549314, 1923.5307517883625, 1923.2834764105216, 1923.807670723725, 1923.4750120225149, 1923.369686772426, 1923.156345672176, 1923.465590027002, 1923.62640167088, 1923.3088046580465, 1923.9133815095624, 1923.9656131442148, 1923.638286279267, 1923.0441240970367, 1923.508831931012, 1923.8765874034746, 1923.736046126117, 1923.992818665909, 1923.3238465552495, 1923.2453472505981, 1923.7069692033076, 1923.3205894187909, 1923.7795053427558, 1923.6228191274236, 1923.1575468870158, 1923.6374642054093, 1923.4785831168867, 1923.671602536396, 1923.5981925212475, 1923.6453316618226, 1923.7649341402403, 1923.7169904621232, 1923.498611370029, 1923.2549646450016, 1923.322678389942, 1923.1880847718362, 1923.3885869312585, 1923.4776291447317, 1923.7244427989287, 1923.487348758799, 1923.6531794363618]
plot_17 = [1926.4830357329138, 1926.4650410855202, 1925.9831404778122, 1926.2647555724388, 1926.605520596334, 1926.5570014056718, 1926.8631464200173, 1926.0801191691821, 1926.5724388081276, 1925.78282153051, 1926.5692275014007, 1926.5276202603688, 1926.4512184081016, 1926.194006052419, 1926.5129954999043, 1926.6396040848858, 1926.6261531996322, 1926.033049032335, 1926.6530730636991, 1926.5368688799933, 1926.570939388651, 1926.4942445940444, 1926.4900488773658, 1926.3878200908273, 1926.722777608572, 1926.8742851332884, 1926.3194177276623, 1926.0636906202046, 1926.3961916337412, 1926.7004351605146, 1926.109317534664, 1926.216413983822, 1926.765093405416, 1926.1880095954616, 1926.1829037757234, 1926.4797570839992, 1926.8085024349798, 1926.0717561011716, 1926.5594832325637, 1926.1399125426626, 1926.813048694064, 1926.6780272990559, 1926.353351504114, 1926.539728312316, 1926.3501572283503, 1926.0045825678217, 1926.6866366282702, 1926.3401409655826, 1926.503211276341, 1926.2990873529236]

plot_nested_list = [plot_0,
                    plot_1,
                    #plot_2,
                    plot_3,
                    #plot_4,
                    plot_5,
                    plot_6,
                    plot_7,
                    plot_8,
                    plot_9,
                    plot_10,
                    plot_11,
                    plot_12,
                    plot_13,
                    plot_14,
                    plot_15,
                    plot_16,
                    plot_17]



x_axis_x = np.linspace(-.7, .7)
x_axis_y = [0] * len(x_axis_x)

y_axis_y = np.linspace(-1, 1)
y_axis_x = [0] * len(y_axis_y)

# %% Stats

#mean_int = np.mean(integral_list)
#std_int = np.std(integral_list)

mean_list = []
std_list = []

for plot in plot_nested_list:
    mean_list.append(np.mean(plot))
    std_list.append(np.std(plot))


# %% Plotting

#fig1 = plt.figure(figsize=(14, 8))
fig1 = plt.figure(figsize=(10, 6))

ax1 = fig1.add_subplot(1, 2, 1)
ax1.plot(x_axis_x, x_axis_y, 'k--', alpha=0.5)
ax1.plot(y_axis_x, y_axis_y, 'k--', alpha=0.5)
ax1.scatter(swe_ang_list, col_ang_list)
ax1.set(title='Mean Center Location', xlabel='Sweep Angle (deg)', ylabel='Probe Angle (deg)')
plt.xlim([-.7, .7])
plt.ylim([-1, 1])
ax1.grid(True)

ax2 = fig1.add_subplot(1, 2, 2)
ax2.scatter(mean_list, std_list)
ax2.set(title='Mean vs. Standard Deviation', xlabel='Mean', ylabel='Standard Deviation')
#plt.xlim([-.7, .7])
plt.ylim([0, 1.4])
ax2.grid(True)


# %%
plt.tight_layout()
plt.show()

