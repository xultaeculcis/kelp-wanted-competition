from matplotlib.colors import ListedColormap

TRAIN = "train"
VAL = "val"
TEST = "test"

SPLITS = [TRAIN, VAL, TEST]

EPS = 1e-10
BACKGROUND = "background"
KELP = "kelp"

CLASSES = [BACKGROUND, KELP]
NUM_CLASSES = len(CLASSES)

ORIGINAL_BANDS = ["SWIR", "NIR", "R", "G", "B", "QA", "DEM"]

CMAP = ListedColormap(["black", "lightseagreen"])

DATASET_STATS = {
    "SWIR": {
        "mean": 9643.5576171875,
        "std": 4047.99609375,
        "min": 0.0,
        "max": 65535.0,
        "q01": 6329.2734375,
        "q99": 17833.78515625,
    },
    "NIR": {
        "mean": 10324.068359375,
        "std": 4853.8349609375,
        "min": 0.0,
        "max": 65535.0,
        "q01": 6478.66943359375,
        "q99": 19959.033203125,
    },
    "R": {
        "mean": 8396.0107421875,
        "std": 2749.9033203125,
        "min": 0.0,
        "max": 65535.0,
        "q01": 6376.58154296875,
        "q99": 13936.0263671875,
    },
    "G": {
        "mean": 8512.5693359375,
        "std": 2710.48046875,
        "min": 0.0,
        "max": 65535.0,
        "q01": 6536.88818359375,
        "q99": 13667.4052734375,
    },
    "B": {
        "mean": 8206.05859375,
        "std": 2633.087890625,
        "min": 0.0,
        "max": 65535.0,
        "q01": 6340.12841796875,
        "q99": 12672.953125,
    },
    "QA": {
        "mean": 0.0,
        "std": 1.0,
        "min": 0.0,
        "max": 1.0,
        "q01": 0.0,
        "q99": 1.0,
    },
    "DEM": {
        "mean": 14.43360710144043,
        "std": 39.570106506347656,
        "min": 0.0,
        "max": 706.0,
        "q01": 0.05749893561005592,
        "q99": 98.20318603515625,
    },
    "ATSAVI": {
        "mean": -0.03758559376001358,
        "std": 0.13958509266376495,
        "min": -1.4883720874786377,
        "max": 0.9999951124191284,
        "q01": -0.16958029568195343,
        "q99": 0.2297411412000656,
    },
    "AFRI1600": {
        "mean": 10323.705078125,
        "std": 4853.77685546875,
        "min": -1.0,
        "max": 65534.9375,
        "q01": 6478.322265625,
        "q99": 19958.677734375,
    },
    "AVI": {
        "mean": 12252.1181640625,
        "std": 7731.0654296875,
        "min": -53256.0,
        "max": 126984.0,
        "q01": 5963.62548828125,
        "q99": 27922.638671875,
    },
    "ARVI": {
        "mean": -0.09480727463960648,
        "std": 0.13175776600837708,
        "min": -1.3499999046325684,
        "max": 0.9899999499320984,
        "q01": -0.215359166264534,
        "q99": 0.16389791667461395,
    },
    "BWDRVI": {
        "mean": -0.7459387183189392,
        "std": 0.17471003532409668,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.8339569568634033,
        "q99": -0.5344080924987793,
    },
    "ClGreen": {
        "mean": 1513730176.0,
        "std": 242724388864.0,
        "min": -1.0,
        "max": 341390005370880.0,
        "q01": -0.22481641173362732,
        "q99": 9054024704.0,
    },
    "CVI": {
        "mean": 380981575680.0,
        "std": 182922522918912.0,
        "min": 0.0,
        "max": 8.246343393072906e17,
        "q01": 0.7174423336982727,
        "q99": 899910008832.0,
    },
    "CI": {
        "mean": -10806794.0,
        "std": 25195399168.0,
        "min": -80009997844480.0,
        "max": 1.0,
        "q01": -0.1364235281944275,
        "q99": 0.18603432178497314,
    },
    "GDVI": {
        "mean": 1811.5123291015625,
        "std": 3530.007568359375,
        "min": -59738.0,
        "max": 63548.0,
        "q01": -1349.5123291015625,
        "q99": 8854.755859375,
    },
    "DVIMSS": {
        "mean": 16381.791015625,
        "std": 9634.6494140625,
        "min": -50816.3984375,
        "max": 153198.0,
        "q01": 8583.703125,
        "q99": 35815.5234375,
    },
    "EVI": {
        "mean": 0.5907021164894104,
        "std": 52.64213562011719,
        "min": -20000.0,
        "max": 20000.0,
        "q01": -2.5075466632843018,
        "q99": 4.32862663269043,
    },
    "EVI2": {
        "mean": 0.17474737763404846,
        "std": 0.2702617645263672,
        "min": -2.399672269821167,
        "max": 2.3999292850494385,
        "q01": -0.07252790778875351,
        "q99": 0.7054058313369751,
    },
    "EVI22": {
        "mean": 0.11927548795938492,
        "std": 0.18636472523212433,
        "min": -1.04160737991333,
        "max": 2.4999260902404785,
        "q01": -0.041537970304489136,
        "q99": 0.5015674233436584,
    },
    "GARI": {
        "mean": 34932576.0,
        "std": 110008336384.0,
        "min": -1165239928225792.0,
        "max": 96899998679040.0,
        "q01": -4.230295181274414,
        "q99": 4.677271842956543,
    },
    "GNDVI": {
        "mean": 0.06501901894807816,
        "std": 0.12524935603141785,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.056693412363529205,
        "q99": 0.2981167733669281,
    },
    "GRNDVI": {
        "mean": -0.2526407241821289,
        "std": 0.12556998431682587,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.36869850754737854,
        "q99": -0.028491679579019547,
    },
    "GBNDVI": {
        "mean": -0.2472846955060959,
        "std": 0.14316342771053314,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.3778710961341858,
        "q99": 0.0018742169486358762,
    },
    "GVMI": {
        "mean": 0.051789261400699615,
        "std": 0.13639113306999207,
        "min": -0.9999814033508301,
        "max": 0.8187726140022278,
        "q01": -0.05805671587586403,
        "q99": 0.21686241030693054,
    },
    "IPVI": {
        "mean": -75986368.0,
        "std": 26661394432.0,
        "min": -40074997661696.0,
        "max": 22898.265625,
        "q01": 1638.6385498046875,
        "q99": 3831.1875,
    },
    "I": {
        "mean": 823.4314575195312,
        "std": 253.89712524414062,
        "min": 3.278688301758259e-12,
        "max": 6446.0654296875,
        "q01": 636.522705078125,
        "q99": 1305.6864013671875,
    },
    "H": {
        "mean": 0.6361790895462036,
        "std": 1.397927165031433,
        "min": -1.5707963705062866,
        "max": 1.5707963705062866,
        "q01": -1.543042540550232,
        "q99": 1.5707812309265137,
    },
    "LogR": {
        "mean": -0.7866759896278381,
        "std": 4.591656684875488,
        "min": -23.025850296020508,
        "max": 33.45589828491211,
        "q01": -3.2276134490966797,
        "q99": 0.6188013553619385,
    },
    "MVI": {
        "mean": 1.0175279378890991,
        "std": 0.2736048698425293,
        "min": 0.0,
        "max": 10.03587818145752,
        "q01": 0.7723481059074402,
        "q99": 1.392896056175232,
    },
    "MSAVI": {
        "mean": 0.11627986282110214,
        "std": 0.19517481327056885,
        "min": -120.513427734375,
        "max": 1.0001220703125,
        "q01": -0.09456334263086319,
        "q99": 0.4435231685638428,
    },
    "NLI": {
        "mean": 0.9591567516326904,
        "std": 0.19729584455490112,
        "min": -1.0,
        "max": 1.0,
        "q01": 0.8619550466537476,
        "q99": 0.9999337792396545,
    },
    "NDVI": {
        "mean": 0.07281439006328583,
        "std": 0.11261355876922607,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.030221495777368546,
        "q99": 0.2939298450946808,
    },
    "NDVIWM": {
        "mean": 0.0,
        "std": 1.0,
        "min": 0.0,
        "max": 1.0,
        "q01": 0.0,
        "q99": 1.0,
    },
    "NDWI": {
        "mean": 0.06501901894807816,
        "std": 0.12524935603141785,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.056693412363529205,
        "q99": 0.2981167733669281,
    },
    "NDWIWM": {
        "mean": 0.0,
        "std": 1.0,
        "min": 0.0,
        "max": 1.0,
        "q01": 0.0,
        "q99": 1.0,
    },
    "NormG": {
        "mean": 0.3057790696620941,
        "std": 0.07158458232879639,
        "min": 0.0,
        "max": 1.0,
        "q01": 0.22100768983364105,
        "q99": 0.35732007026672363,
    },
    "NormNIR": {
        "mean": 0.35333243012428284,
        "std": 0.0925254300236702,
        "min": 0.0,
        "max": 1.0,
        "q01": 0.2720213234424591,
        "q99": 0.4808278977870941,
    },
    "NormR": {
        "mean": 0.3001948893070221,
        "std": 0.06703205406665802,
        "min": 0.0,
        "max": 0.9553191661834717,
        "q01": 0.2225407212972641,
        "q99": 0.3389860689640045,
    },
    "PNDVI": {
        "mean": -0.4159976541996002,
        "std": 0.139930859208107,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.5304610133171082,
        "q99": -0.18218237161636353,
    },
    "SRGR": {
        "mean": 163926208.0,
        "std": 57107148800.0,
        "min": 0.0,
        "max": 91189999042560.0,
        "q01": 0.8017561435699463,
        "q99": 1.1494911909103394,
    },
    "SRNIRG": {
        "mean": 1513730176.0,
        "std": 242724388864.0,
        "min": 0.0,
        "max": 341390005370880.0,
        "q01": 0.7751835584640503,
        "q99": 9054024704.0,
    },
    "SRNIRR": {
        "mean": 1461456000.0,
        "std": 223439912960.0,
        "min": 0.0,
        "max": 338619986345984.0,
        "q01": 0.8166563510894775,
        "q99": 3979606016.0,
    },
    "SRNIRSWIR": {
        "mean": 1.0175279378890991,
        "std": 0.2736048698425293,
        "min": 0.0,
        "max": 10.03587818145752,
        "q01": 0.7723481059074402,
        "q99": 1.392896056175232,
    },
    "SRSWIRNIR": {
        "mean": 344121216.0,
        "std": 152180539392.0,
        "min": 0.0,
        "max": 106959994880000.0,
        "q01": 0.6507546305656433,
        "q99": 1.139648199081421,
    },
    "RBNDVI": {
        "mean": -0.2446092814207077,
        "std": 0.13592885434627533,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.36583688855171204,
        "q99": -0.0015550563111901283,
    },
    "SQRTNIRR": {
        "mean": 270.6546325683594,
        "std": 38228.03515625,
        "min": 0.0,
        "max": 18401630.0,
        "q01": 0.837624192237854,
        "q99": 752.1063842773438,
    },
    "TNDVI": {
        "mean": 0.7534251809120178,
        "std": 0.07217277586460114,
        "min": 0.01904531568288803,
        "max": 1.2247449159622192,
        "q01": 0.687389612197876,
        "q99": 0.8889976143836975,
    },
    "TVI": {
        "mean": 0.7013707756996155,
        "std": 0.02283400110900402,
        "min": 0.028014538809657097,
        "max": 1.2247449159622192,
        "q01": 0.6737876534461975,
        "q99": 0.7338758111000061,
    },
    "VARIGreen": {
        "mean": -288136.3125,
        "std": 7342396416.0,
        "min": -28660000096256.0,
        "max": 152410001506304.0,
        "q01": -0.07021640241146088,
        "q99": 0.11139820516109467,
    },
    "WDRVI": {
        "mean": -0.7545422315597534,
        "std": 0.16317947208881378,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.8270394802093506,
        "q99": -0.5871533751487732,
    },
    "DEMWM": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "q01": 0.0, "q99": 1.0},
    "TURB": {
        "mean": 225674854400.0,
        "std": 24313212174336.0,
        "min": -6.389999866485596,
        "max": 5852275729760256.0,
        "q01": 0.8975217938423157,
        "q99": 6091393990656.0,
    },
    "CDOM": {
        "mean": 48.392112731933594,
        "std": 100.96046447753906,
        "min": 0.0,
        "max": 537.0,
        "q01": 21.974166870117188,
        "q99": 107.1183853149414,
    },
    "DOC": {
        "mean": 60.36745834350586,
        "std": 76.91044616699219,
        "min": 0.0,
        "max": 432.0,
        "q01": 37.36685562133789,
        "q99": 107.90576171875,
    },
    "WATERCOLOR": {
        "mean": 1286.4730224609375,
        "std": 4968.59521484375,
        "min": 0.0,
        "max": 25366.0,
        "q01": 183.43519592285156,
        "q99": 3946.123046875,
    },
    "SABI": {
        "mean": 1432078464.0,
        "std": 227292168192.0,
        "min": -7349999763456.0,
        "max": 339339997347840.0,
        "q01": -0.03082076832652092,
        "q99": 8786300928.0,
    },
    "KIVU": {
        "mean": -71205768.0,
        "std": 28124923904.0,
        "min": -48729998491648.0,
        "max": 48089998032896.0,
        "q01": -256337616.0,
        "q99": 0.12153837084770203,
    },
    "Kab1": {
        "mean": 0.6200457811355591,
        "std": 2.66886305809021,
        "min": -120.70118713378906,
        "max": 134.31333923339844,
        "q01": -0.026979811489582062,
        "q99": 2.3947947025299072,
    },
    "NDAVI": {
        "mean": 0.08156391978263855,
        "std": 0.15248508751392365,
        "min": -1.0,
        "max": 1.0,
        "q01": -0.052819494158029556,
        "q99": 0.35587695240974426,
    },
    "WAVI": {
        "mean": 0.12234315276145935,
        "std": 0.22872214019298553,
        "min": -1.4999017715454102,
        "max": 1.4999885559082031,
        "q01": -0.07922711968421936,
        "q99": 0.5338025689125061,
    },
}
