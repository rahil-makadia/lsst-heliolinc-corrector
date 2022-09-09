### Python Differential Corrector Integrator
"""
dc_integrator

A propagator for the Python Differential Corrector for LSST HelioLinc3D results
Implementation: Python 3.8, R. Makadia 09092022
"""

import numpy as np
import spiceypy as sp
from spiceypy import ctypes, libspice, Tuple, ndarray, Union, Iterable
from numba import jit

############################################
# Global Variables
###########################################
# list of spice kernels. DE4xx planetary ephemeris must come first.
spkFile = ['data/planets_big16_de441_1950_2350.bsp']

array = np.array
ascontiguousarray = np.ascontiguousarray
concat = np.concatenate
zeros = np.zeros
norm = np.linalg.norm
cross = np.cross
dot = np.dot
sin = np.sin
cos = np.cos
float64 = np.float64

############################################
# MODULE SPECIFIC EXCEPTION
###########################################
class Error(Exception):
    """Module specific exception."""
    pass

############################################
# Functions
###########################################
#### custom spiceypy functions
def spkez_mod(targ: list, et: float, ref: str, abcorr: str, obs: int) -> Tuple[ndarray, float]:
    """
    Return the state (position and velocity) of a target body
    relative to an observing body, optionally corrected for light
    time (planetary aberration) and stellar aberration.

    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkez_c.html

    :param targ: Target body.
    :param et: Observer epoch.
    :param ref: Reference frame of output state vector.
    :param abcorr: Aberration correction flag.
    :param obs: Observing body.
    :return:
            State of target,
            One way light time between observer and target.
    """
    
    et = ctypes.c_double(et)
    ref = bytes(ref, encoding='utf-8') # stypes.string_to_char_p(ref)
    abcorr = bytes(abcorr, encoding='utf-8') # stypes.string_to_char_p(abcorr)
    obs = ctypes.c_int(obs)
    lt = ctypes.c_double()

    def return_single_state(target):
        starg = (ctypes.c_double * 6)()
        libspice.spkez_c(ctypes.c_int(target), et, ref, abcorr, obs, starg, lt)
        return starg
    states = [return_single_state(target) for target in targ]
    return states

def spkezr_mod(
    targ: str, et: Union[ndarray, float], ref: str, abcorr: str, obs: str
) -> Union[Tuple[ndarray, float], Tuple[Iterable[ndarray], Iterable[float]]]:
    """
    Return the state (position and velocity) of a target body
    relative to an observing body, optionally corrected for light
    time (planetary aberration) and stellar aberration.

    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html

    :param targ: Target body name.
    :param et: Observer epoch.
    :param ref: Reference frame of output state vector.
    :param abcorr: Aberration correction flag.
    :param obs: Observing body name.
    :return:
            State of target,
            One way light time between observer and target.
    """
    targ = bytes(targ, encoding='utf-8') # stypes.string_to_char_p(targ)
    ref = bytes(ref, encoding='utf-8') # stypes.string_to_char_p(ref)
    abcorr = bytes(abcorr, encoding='utf-8') # stypes.string_to_char_p(abcorr)
    obs = bytes(obs, encoding='utf-8') # stypes.string_to_char_p(obs)
    lt = ctypes.c_double()

    if hasattr(et, "__iter__"):
        def return_single_state(time):
            starg = (ctypes.c_double * 6)()
            libspice.spkezr_c(targ, ctypes.c_double(time), ref, abcorr, obs, starg, lt)
            return starg
        states = [return_single_state(time) for time in et]
    else:
        starg = (ctypes.c_double * 6)()
        libspice.spkezr_c(targ, ctypes.c_double(et), ref, abcorr, obs, starg, lt)
        states = starg
    return states

def getDEpos(target, t, frame='ECLIPJ2000'):
    """
    Get a target body's position at an epoch in the specified frame from the DE epehemerides

    Parameters:
    -----------
    target ... Target body NAIF ID
    t      ... Query time (in NAIF SPICE Ephemeris time)
    frame  ... Co-ordinate frame for desired position

    Returns:
    --------
    pos    ... Position of the target body at query time in specified frame
    """

    return sp.spkezp(target, t, frame, 'NONE', 0)[0]


def getDEstate(target, t, frame='ECLIPJ2000'):
    """
    Get a target body's state at an epoch in the specified frame from the DE epehemerides

    Parameters:
    -----------
    target ... Target body NAIF ID
    t      ... Query time (in NAIF SPICE Ephemeris time)
    frame  ... Co-ordinate frame for desired state

    Returns:
    --------
    state  ... State of the target body at query time in specified frame
    """

    return sp.spkez(target, t, frame, 'NONE', 0)[0]


def loadSpiceKernels(spkfiles=spkFile):
    """
    Load spice kernels. DE4xx planetary ephemeris must come first.

    Parameters:
    -----------
    spkfiles ... Path to spice kernel files

    Returns:
    --------
    none
    """

    for f in spkfiles:
        sp.furnsh(f)

    return None
try:
    loadSpiceKernels(spkFile)
except:
    raise Error('Error: Could not load SPICE Kernels. Please check file path.')


def getEphemerisParameters(filename):
    """
    Get parameters from JPL DExxx ephemeris file.
    Parameters:
    -----------
    filename  ... Path to JPL DExxx file
    Returns:
    --------
    au2km     ... Astronomical unit in kilometers
    clight    ... Light speed
    GMarray      ... Array of G*mass for all perturbing bodies
    Ephemlist    ... List containing the Ephemeris id of perturbers in Ephemorder
    Ephemorder   ... Order in which the accelerations should be accumulated
    ppnlist      ... Consider Parametrized Post Newtonian terms for bodies where ppnlist == True
    """

    au2km = 149597870.7
    clight = 299792.458

    check_string = filename.lower()
    de43x_list = ['de430', 'de431']
    de44x_list = ['de440', 'de441']
    if any(de43x in check_string for de43x in de43x_list):
        # print('DE43x planetary ephemeris constants loaded')
        GMlist = [4.91248045036476e-11, 7.24345233264412e-10, 8.887692445125634e-10, 9.54954869555077e-11, 2.82534584083387e-07, 8.45970607324503e-08, 1.29202482578296e-08, 1.52435734788511e-08, 2.17844105197418e-12, 0.0002959122082855911, 1.093189450742374e-11, 1.400476556172344e-13, 3.104448198938713e-14, 3.617538317147937e-15, 3.85475018780881e-14, 3.748736284552032e-16, 8.312419212673372e-16, 2.136434442571407e-15, 5.894256529706908e-16, 1.07784100424073e-15, 1.235800787294125e-14, 1.331536255459975e-15, 1.93177578518292e-16, 1.797004894507446e-15, 1.100645679575068e-15, 4.678307418350905e-15, 3.411586826193812e-15, 2.081506396469738e-16, 2.008927736651132e-16, 1.035644840131194e-15, 9.199807477630912e-17, 2.529442872040999e-16, 1.20262444348346e-15, 1.895331760419783e-16, 1.893901667525382e-15, 7.239841522366211e-17, 1.637343952261084e-16, 3.888003898545788e-16, 2.926272744294528e-16, 1.97584236512452e-15, 1.482019016437529e-16, 6.343280473648602e-15, 1.19958501623344e-16, 2.944541291521286e-16, 2.352256173241841e-16, 1.697060018409709e-16, 2.185620577113056e-16, 1.323285964746768e-16, 1.497519682556701e-15, 2.952414080308422e-16, 9.324223762198869e-16, 2.765316643474381e-16, 7.275393853340713e-17, 4.68864072012922e-17, 8.425678018567934e-16, 3.2728e-16, 5.543523561598889e-16, 2.531091726015068e-15, 7.549481629314401e-17, 1.633326391117518e-16, 2.570549113353145e-16, 2.476788101255867e-15, 6.239243310775165e-17, 5.624173650192459e-16, 3.699288312702126e-16, 3.680601920639651e-16, 8.481173911466002e-17, 6.339442727587651e-16, 5.091136783014464e-17, 1.089048191960057e-16, 5.640401743976243e-17, 3.180659282652541e-15, 3.431026591237969e-16, 5.144610020876735e-16, 2.768888840157846e-16, 1.424492746350956e-16, 7.995051044916541e-17, 3.507374451295614e-16, 4.357374625077127e-17, 8.312200000000001e-16, 4.931295509500729e-17, 8.401906253463887e-17, 8.35182433140794e-17, 1.161443954113108e-16, 1.022367554556134e-16, 6.60126076693077e-17, 1.096834890626035e-16, 1.257312655631886e-16, 9.254085453018537e-16, 2.152399557022891e-16, 2.199295173574073e-15, 2.577114127311047e-15, 3.402031157439429e-16, 1.235196362828491e-16, 2.440461677701006e-16, 4.036943517686073e-16, 5.647730717976476e-16, 1.27923e-15, 2.716619708393259e-16, 1.546567695624325e-15, 1.031495635837631e-16, 2.442831741732069e-16, 7.352662013841591e-17, 1.365977196468697e-16, 1.281266225660598e-16, 1.001118016858646e-16, 4.812239667801873e-16, 3.715466797534145e-16, 5.397399999999998e-16, 1.671720991700644e-15, 1.082618586158193e-16, 1.407698572210504e-16, 3.351919281128056e-17, 5.796039701553235e-17, 2.558021392457819e-17, 1.705e-16, 5.525824190385527e-17, 4.471368017841789e-16, 2.700075962598135e-17, 1.188984949952008e-15, 7.007906922041343e-16, 8.877270826562338e-17, 3.661168243061723e-16, 9.650129510548752e-16, 4.654245247397975e-16, 9.936629545909249e-16, 1.319614122170156e-17, 3.362046542088716e-16, 9.515605041846208e-17, 8.561260599553892e-16, 4.224288214377445e-16, 3.131673253222406e-16, 3.766140601670833e-16, 1.126086114748883e-16, 6.995163354983087e-16, 7.558232926228892e-16, 3.9416e-16, 4.068172917716431e-16, 1.65952595163453e-16, 4.533646848796812e-16, 8.298388767163694e-16, 2.6336e-16, 2.8824066183805e-16, 1.024848705874405e-16, 2.372907537934201e-16, 8.658e-17, 4.18326020317249e-16, 1.068459902704693e-15, 5.347099999999998e-16, 1.510988490929355e-16, 4.317715764492053e-17, 3.912428547083101e-16, 2.899097196211582e-16, 1.671347552009361e-16, 4.537288629425099e-17, 1.679676683354825e-16, 1.135585894483922e-15, 9.41936446359531e-16, 1.725255523638289e-16, 2.511941265657833e-16, 2.723058728905982e-16, 8.354819850737808e-17, 4.501413656117928e-16, 2.053857682908331e-17, 1.755461678252614e-16, 1.014729689571656e-16, 1.849247208030054e-16, 9.151499922595894e-17, 1.178966984935731e-16, 2.593107309887293e-16, 1.31453781488021e-16, 3.04651155649041e-16, 3.867499657751679e-16, 9.522389187217239e-17, 6.907971247467425e-16, 1.816840620173303e-16, 1.390603968519846e-16, 3.952059034345272e-17, 4.145657172535278e-16, 1.204845954546502e-16, 1.760898715515135e-16, 1.971591966625455e-16, 1.136329390113381e-16, 5.296669358073911e-16, 1.104862252873265e-16, 3.005483625946005e-16, 2.709727072416213e-16, 1.968855018599242e-16, 6.280858553936383e-16, 1.841330187523782e-16, 5.266901109254348e-16, 1.8654e-16, 2.248155048843683e-16, 2.052810177986958e-16, 4.80337378560961e-17, 2.884868403951011e-16, 6.883271096605401e-17, 7.240547885258139e-16, 1.158459133740796e-16, 8.276349521304358e-17, 1.388626589856199e-15, 1.891524746656209e-16, 2.9127e-16, 5.114309394296057e-17, 3.262493379434562e-16, 1.475676925888911e-16, 5.643e-17, 3.451507451186596e-17, 5.14076834286322e-17, 5.368909704258335e-16, 1.231752116988413e-16, 2.209916607717656e-16, 2.194912217781737e-17, 7.007873927130292e-16, 1.410007840576309e-16, 1.585098657159689e-15, 2.681613196135182e-16, 1.761956026100257e-16, 1.217809799583179e-16, 3.367760574353007e-16, 8.087079288103283e-17, 6.984194530211866e-17, 7.7695552096218e-17, 1.156395471364583e-16, 5.734581566336858e-17, 1.7459557262705e-15, 1.372061759613779e-16, 4.558599248806925e-16, 1.414422892089808e-16, 3.593493480761761e-16, 1.103185182296828e-16, 1.507933371196519e-15, 1.004465983963094e-16, 3.433344412859169e-16, 6.225101838738033e-17, 1.5586e-16, 1.452167895484219e-16, 5.576480476808539e-16, 1.567850414047483e-16, 8.214499999999999e-16, 3.618654697029734e-16, 1.465573497563191e-16, 5.400121408743423e-17, 1.486985629344961e-16, 3.6782e-16, 4.8352e-16, 2.112438360599952e-15, 7.379985344375229e-17, 2.297975812797145e-16, 6.340778349521697e-17, 1.878012295898928e-17, 9.900001189795902e-19, 5.553465628746912e-17, 9.070804844114505e-16, 1.153421084931585e-16, 1.355817130673488e-16, 2.295559390637462e-15, 7.680884774699999e-17, 2.28615933395781e-16, 8.063518429005578e-17, 6.539925772440235e-17, 2.707141673652781e-16, 3.2284e-16, 8.459430728959683e-16, 2.419316056464066e-16, 3.340700529704519e-16, 3.673796079923418e-17, 1.915156279885078e-16, 5.484672429113171e-16, 3.641497397783291e-16, 1.485897083825289e-16, 1.298792923917022e-16, 1.118718331450076e-16, 3.417698501747239e-16, 1.973531391104841e-16, 3.410676979753487e-16, 5.198126979457498e-15, 2.937652079531482e-16, 6.96e-17, 1.08862665088002e-16, 1.812449396445861e-16, 9.31594859406562e-16, 7.253861643466114e-17, 1.097563103282251e-15, 1.948498611557125e-16, 2.812357745865768e-17, 2.74856688034015e-16, 6.2674e-16, 8.597802188183131e-17, 6.112321972668314e-17, 2.176059561415946e-17, 1.17169915406794e-17, 2.003839350952163e-17, 5.018899345207492e-17, 2.221113361582845e-16, 3.835089170878002e-16, 6.153064154075524e-17, 6.91187747327401e-17, 2.150146743361225e-16, 6.426039953318643e-17, 4.63868801607939e-16, 1.449797679769329e-17, 1.664377872588215e-16, 1.699566952633139e-16, 3.5953e-16, 1.204954948857818e-16, 9.09429441579076e-17, 1.438374800446789e-16, 1.592347088449227e-16, 9.958162010025519e-17, 1.043610361250656e-16, 4.763753531654247e-16, 9.453705225820102e-17, 1.493866803013356e-16, 1.117773442397485e-16, 8.895067284927045e-16, 5.256168678493662e-15, 2.91750035064825e-16, 1.545600853865991e-16, 5.020424221292565e-16, 1.426815784833301e-16, 8.597620263717324e-17, 7.138243397529348e-17, 1.189281443931769e-16, 6.215746066236688e-16, 1.79893748114765e-16, 4.998204812589248e-17, 5.816792100573278e-17, 2.081801283090566e-16, 1.972515398569376e-16, 2.520614492533895e-16, 1.957616090642333e-16, 3.056711944663653e-16, 4.774354773879983e-17, 1.485740534765631e-16, 1.630823096659809e-16, 1.583019880469992e-16, 2.158155681763601e-16, 1.7558991833247e-15, 1.219648751741075e-16, 2.267103341459905e-16, 4.555959566377237e-16, 1.035497250152759e-16, 3.786475033089484e-16, 3.362975492761039e-16, 4.117268268399542e-17, 1.53529788855662e-16, 1.258376592645005e-16, 7.617140987284265e-17, 6.467767213237461e-17, 2.871640482670196e-16, 9.549128215887647e-17, 6.248568708463761e-17, 1.115280133034817e-16]
        # de43x big16 main belt perturbers from sb431-n16s.bsp
        Ephemorder = ['Cybele', 'Euphrosyne', 'Doris', 'Thisbe', 'Patientia', 'Psyche', 'Juno', 'Eunomia',
                    'Sylvia', 'Europa', 'Interamnia', 'Davida', 'Hygiea', 'Pallas', 'Vesta', 'Ceres',
                    'Pluto', 'Mercury', 'Uranus', 'Neptune', 'Venus', 'Mars', 'Moon', 'Earth', 'Saturn', 'Jupiter', 'Sun']
    elif any(de44x in check_string for de44x in de44x_list):
        # print('DE44x planetary ephemeris constants loaded')
        GMlist = [4.912500194889318e-11, 7.243452332644119e-10, 8.887692446707102e-10, 9.549548829725812e-11, 2.825345825225792e-07, 8.45970599337629e-08, 1.29202656496824e-08, 1.524357347885194e-08, 2.175096464893358e-12, 0.00029591220828411956, 1.093189462402435e-11, 1.396451812308107e-13, 3.04711463300432e-14, 4.282343967799501e-15, 3.8548000225257904e-14, 3.2161933241855116e-16, 1.4423591761343029e-15, 2.5416014973471498e-15, 5.740266647501823e-16, 1.4490138025867273e-15, 1.254253076164081e-14, 1.0258530487431674e-15, 3.2843482824404967e-16, 1.1389436761809467e-15, 1.1863795139948057e-15, 4.5107799051436795e-15, 3.5445002842488978e-15, 1.3398683097840782e-16, 4.818114984817475e-16, 1.2396175182794244e-15, 5.71341843864351e-16, 1.9760028913855683e-16, 8.805494503588215e-16, 2.7673067211701877e-16, 1.3109170444249925e-15, 3.256080020069492e-17, 1.7812616814154862e-16, 2.561386580415216e-16, 3.4043931236013436e-16, 1.778875957037877e-15, 5.705726375789753e-17, 2.4067012218937576e-15, 7.874507166651942e-17, 3.139214460205205e-16, 8.270923167880293e-17, 5.013597383486977e-16, 2.428130550827187e-16, 8.166207221102614e-17, 1.6238080168284406e-15, 3.7393826630015266e-16, 1.2055616051144469e-15, 1.8735257964754667e-16, 1.2683502962561622e-16, 8.119998868610499e-17, 8.037535169551508e-16, 8.292410989683984e-16, 9.548452182565237e-16, 1.908516195648564e-15, 8.030901956158537e-16, 9.065813600941014e-17, 5.596605208562996e-16, 5.982431526486984e-15, 7.709568502565123e-17, 2.561364385673563e-16, 4.3334389285268117e-16, 7.472641289359161e-17, 1.8012505811829385e-16, 6.124244003827881e-16, 1.48260785652372e-17, 1.5213428223941847e-16, 2.859883377373698e-16, 2.0917175955133682e-15, 1.1706570191254942e-16, 1.1215489958719225e-15, 1.221013302136009e-16, 1.3120783938467559e-16, 1.5086969937360234e-16, 1.901171693675278e-16, 5.802745629798349e-17, 7.196534867203428e-16, 5.2724601168331567e-17, 2.0368097558802117e-16, 4.824226918174613e-17, 5.647115427506023e-17, 1.2659277966238143e-16, 3.828565766724855e-17, 1.8459902693767395e-16, 7.495697601430942e-17, 8.410568113447993e-16, 1.436757702345231e-16, 4.834560654610552e-15, 2.6529436610356353e-15, 5.699803244510909e-16, 2.642881600589284e-16, 1.4195955177021533e-16, 2.0222499285918262e-16, 5.890932368757391e-16, 1.9476499609659456e-15, 4.60264620685128e-16, 9.42942121305823e-16, 1.0268048353591737e-16, 3.1864312654754617e-16, 5.014607211190043e-17, 8.550862140934806e-17, 7.271262098124241e-17, 1.2873552745693193e-16, 4.4137261924065115e-16, 1.404934369679812e-16, 5.933985070091769e-17, 3.219139207587859e-15, 8.73429262293014e-17, 1.127327762395292e-16, 1.2685989278560462e-16, 3.051962645845043e-17, 3.081992520561064e-17, 2.488406700654089e-16, 6.973540602439245e-17, 8.795632662491299e-16, 1.3157171164548998e-17, 1.0049283867977807e-15, 3.002775558113526e-16, 1.4082516742158776e-16, 2.475490134670677e-16, 8.584786131141037e-16, 1.3694441284769087e-16, 1.3272835717079433e-15, 1.7504352843147015e-17, 1.2041493785248888e-16, 1.4516053346827936e-16, 2.6291667570900683e-16, 9.259555216971151e-16, 3.345175275226785e-16, 2.1889333845457843e-16, 2.49831855261886e-16, 4.073838070777593e-16, 3.131949350342742e-16, 7.414551782088354e-16, 2.485857842368132e-16, 2.7635117872432285e-16, 2.690665426465701e-16, 1.9062882233624486e-15, 2.7907996428036474e-16, 1.9236219019969684e-16, 6.514251331550227e-17, 9.901268916551379e-17, 1.350142511906315e-16, 2.685636326205827e-16, 2.645429961552526e-16, 8.286215123097553e-16, 3.026253130619937e-16, 4.658827122108877e-17, 3.494445297018901e-16, 7.312424159554999e-17, 1.5731944639940773e-16, 3.194885761351898e-17, 5.123339897726351e-16, 1.4541092712549263e-16, 6.998876221714286e-16, 9.270050959529838e-17, 8.142129140295343e-17, 2.4561528144043314e-16, 8.831697737909073e-17, 7.188369171784315e-16, 3.4690964938059876e-17, 1.506374322665671e-16, 1.6663045771433504e-16, 5.180479043790148e-16, 5.23643121338054e-17, 1.7485910807435368e-16, 1.951100813450661e-15, 7.66348632340929e-17, 7.560179850744221e-17, 4.460510689200354e-16, 7.281410210178525e-17, 4.142728844408081e-16, 1.4535063882245356e-16, 5.521083382556173e-17, 4.638450322774714e-17, 6.148848913653127e-17, 5.294972789322552e-16, 2.6546896309806863e-16, 1.997962433107355e-16, 6.25172630260285e-17, 2.2188561497353161e-16, 9.25549528015023e-17, 8.257891885518958e-16, 7.216647312452579e-16, 4.69449662479354e-16, 4.525108943500125e-16, 1.6594282580293835e-16, 3.8623042613708653e-16, 1.256041563617464e-16, 1.9922195146137866e-16, 3.8354815246046577e-16, 5.1065425768005637e-17, 1.2250630736040161e-16, 8.00596320733248e-17, 2.1647804370959252e-16, 7.187570798689379e-17, 8.072837727772762e-17, 1.3808681586222395e-15, 6.832658092241454e-17, 4.998246830863383e-16, 5.022579920320677e-17, 2.413622033962929e-16, 2.084313266700059e-16, 1.4252117646877083e-17, 7.310312768200536e-17, 2.864982709174516e-17, 1.3166985822586556e-16, 1.5891479446112133e-16, 1.0985328629139597e-16, 2.4842140182945907e-17, 7.777039207208923e-16, 2.7857999144799436e-16, 6.979942758868987e-16, 2.5189815204630036e-16, 2.8721263188047396e-16, 9.670798400851406e-17, 2.0487987640827176e-16, 1.1276722765420132e-16, 2.239212640831399e-16, 8.439600858609733e-17, 7.978946832542661e-17, 8.336782410225417e-17, 1.2340564359220072e-15, 1.2907423434479045e-16, 1.771687529490865e-15, 9.303415960975477e-17, 4.49332166347459e-16, 9.976185628516443e-17, 1.6744520343182799e-15, 3.438304449717668e-16, 2.6866209145075665e-16, 8.005933925741988e-17, 3.3572279402433382e-16, 1.6144760474032773e-16, 2.834344202759018e-16, 1.215964969184977e-16, 9.307130180298934e-16, 3.152731937031252e-16, 1.2572594904492957e-16, 8.057307747567055e-17, 1.6511852151088483e-16, 2.377357820920586e-16, 2.3670178521081795e-16, 1.1511452195400677e-15, 1.1020156825676931e-16, 3.2507694999797983e-16, 1.398133673348538e-16, 1.3601888558172924e-17, 9.950600155464848e-19, 3.7681035098461244e-17, 4.24710309935973e-16, 8.531561503523623e-17, 1.138171532831636e-16, 1.2973797046097596e-15, 6.679819241829383e-17, 1.1481333345022554e-16, 5.823302640284164e-17, 5.186719528510211e-17, 7.528350655020485e-17, 3.6478450110028057e-16, 1.0154946317000345e-15, 9.329084948707906e-16, 1.4232478918230183e-16, 4.5881147721288083e-17, 7.952288319351003e-16, 2.8775529812775966e-16, 1.4268068051088923e-16, 9.032721399920834e-17, 1.6166164033700236e-16, 8.207675177228163e-17, 1.1434524745954757e-16, 1.9069839846057659e-16, 2.600374703356517e-16, 8.683625349228654e-15, 1.886129575533859e-16, 5.663027638551713e-17, 1.765592961183617e-16, 1.558411364789865e-16, 1.765129565112637e-15, 6.050663786253968e-17, 2.7333272315776016e-16, 1.9519850083243681e-16, 3.289014852337975e-17, 1.5935521711557882e-16, 1.7007818925283807e-15, 7.974665095828324e-17, 5.56926103195199e-17, 3.8849243156686706e-17, 9.502726030864262e-18, 1.5650082930549903e-17, 4.3152660225791144e-17, 2.2465030240169206e-16, 8.750385462374579e-16, 9.203383666159985e-17, 1.0123777316180206e-16, 2.280967490901478e-16, 4.6458511951546986e-17, 2.519320488920722e-16, 1.0120911587092832e-17, 7.02207690642838e-17, 1.0924383869373906e-16, 1.827115384502823e-16, 1.9725537729863713e-16, 1.7174750596131368e-16, 9.935357257488864e-17, 1.0490232752479031e-16, 8.150019061382226e-17, 1.2001846940536772e-16, 2.5859089556681753e-16, 5.120974853484182e-17, 5.1009798048889617e-17, 6.773715668991604e-17, 1.6730262398441171e-15, 6.311034342087889e-15, 3.9116752395114746e-16, 1.193204585427415e-16, 1.2600845433723963e-16, 1.114915570450872e-16, 7.565550192634045e-17, 1.189570423520968e-16, 1.1508271223874676e-16, 8.916216061479975e-16, 1.5375064738104504e-16, 5.5029875850101036e-17, 6.805013604111056e-17, 4.114622959542111e-16, 1.2538556013158105e-16, 1.736544758134813e-16, 1.1496018031508255e-16, 6.914477361926693e-16, 2.0592519759347046e-17, 5.683773262812121e-16, 5.318933742144326e-17, 1.9534428745516884e-16, 1.771187154994267e-16, 5.043148194735074e-16, 2.1076085204803308e-16, 2.552783047004986e-16, 5.745072085427062e-17, 9.51207551986228e-17, 1.8942493318435747e-16, 3.964693261446294e-17, 4.449283120191586e-17, 6.347027638446878e-17, 1.477401852453462e-16, 8.020428649722863e-17, 2.112761621018509e-17, 1.3774121589661637e-16, 1.2924733629447915e-16, 1.3076632020262883e-16, 9.53947556947391e-17, 2.485448030287316e-12, 5.961630054504284e-13, 1.4934867884244863e-13, 3.431855500629209e-13, 1.5294877459751145e-13, 2.281996343120454e-13, 9.409624643156109e-14, 5.495075223411865e-14, 1.8603512445448687e-14, 4.517665622509759e-14, 7.653501749067198e-14, 3.965776886964191e-14, 6.991602589737882e-14, 2.904099518188484e-14, 6.02473242459976e-14, 3.6285479917789626e-14, 3.719350237382443e-14, 6.082123528394769e-14, 2.585678827709039e-13, 2.538379645629731e-14, 6.511702079363127e-14, 2.67691589753377e-14, 2.9579875156960054e-14, 3.645022602009109e-15, 2.7251938920337723e-14, 6.483858075581723e-14, 1.7561425934907343e-14, 1.912886696869436e-14, 2.1338903510045792e-14, 2.571034257081021e-14, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13, 5.522769971698821e-13]
        # de44x big16 main belt perturbers from sb441-n16s.bsp
        Ephemorder = ['Cybele', 'Euphrosyne', 'Iris', 'Thisbe', 'Camilla', 'Psyche', 'Juno', 'Eunomia',
                    'Sylvia', 'Europa', 'Interamnia', 'Davida', 'Hygiea', 'Pallas', 'Vesta', 'Ceres',
                    'Pluto', 'Mercury', 'Uranus', 'Neptune', 'Venus', 'Mars', 'Moon', 'Earth', 'Saturn', 'Jupiter', 'Sun']
    else:
        raise Error('NAIF planetary DE4xx ephemeris file not recognized. ' \
                        'Please make sure that the planetary ephemeris file is the first entry in spk_files, not a minor body ephemeris. ' \
                        f'Your ephemeris version might not currently be supported. Currently supported files are: {de43x_list+de44x_list}')

    # SPICE IDs for solar system planets and perturbing asteroids
    Ephemdict = {'Sun': 10, 'Mercury': 1, 'Venus': 2, 'Earth': 399, 'Moon': 301,
                    'Mars': 4, 'Jupiter': 5,  'Saturn': 6, 'Uranus': 7, 'Neptune': 8,
                    'Pluto': 9, 'Ceres': 2000001, 'Pallas': 2000002, 'Juno': 2000003,
                    'Vesta': 2000004, 'Hebe': 2000006, 'Iris': 2000007, 'Hygiea': 2000010,
                    'Eunomia': 2000015, 'Psyche': 2000016, 'Amphitrite': 2000029,
                    'Europa': 2000052, 'Cybele': 2000065, 'Sylvia': 2000087,
                    'Thisbe': 2000088, 'Eros': 2000433, 'Davida': 2000511,
                    'Interamnia': 2000704, 'Euphrosyne': 2000031, 'Camilla': 2000107,
                    'Doris': 2000048, 'Patientia': 2000451}

    GMdict = {'Sun': 9, 'Mercury': 0, 'Venus': 1, 'Earth': 2, 'Moon': 10,
                'Mars': 3, 'Jupiter': 4,  'Saturn': 5, 'Uranus': 6, 'Neptune': 7,
                'Pluto': 8, 'Ceres': 11, 'Pallas': 12, 'Juno': 13,
                'Vesta': 14, 'Hebe': 16, 'Iris': 17, 'Hygiea': 20,
                'Eunomia': 25, 'Psyche': 26, 'Amphitrite': 39,
                'Europa': 61, 'Cybele': 71, 'Sylvia': 90,
                'Thisbe': 91, 'Eros': 251, 'Davida': 276,
                'Interamnia': 316, 'Euphrosyne': 41, 'Camilla': 109,
                'Doris': 57, 'Patientia': 256}

    Ephemlist = []
    GMarray = []
    for obj in Ephemorder:
        Ephemlist.append(Ephemdict[obj])
        GM = GMlist[GMdict[obj]]
        GMarray.append(GM)
    GMarray = array(GMarray)

    ppnlist = [False]*len(Ephemorder)
    ppnlist[Ephemorder.index('Sun')] = True
    # ppnlist[Ephemorder.index('Jupiter')] = True
    # ppnlist[Ephemorder.index('Earth')] = True

    return au2km, clight, GMarray, Ephemlist, Ephemorder, ppnlist
try:
    au, clight, GMarray, Ephemlist, Ephemorder, ppnlist = getEphemerisParameters(spkFile[0])
except:
    raise Error('Error: Could not acquire Ephemeris parameters. Please check file path for JPL DE ephemeris files.')

@jit('float64(float64)', nopython=True, cache=True)
def mjd2et(MJD):
    """
    Converts modified Julian Date to JPL NAIF SPICE Ephemeris time.
    Only valid for TDB timescales.

    Parameters:
    -----------
    MJD ... Modified Julian Day

    Returns:
    --------
    ET  ... Ephemeris time (ephemeris seconds beyond epoch J2000)
    """

    return (MJD+2400000.5-2451545.0)*86400

@jit('float64(float64)', nopython=True, cache=True)
def jd2mjd(JD):
    """
    Converts Julian Date Modified Julian Date.

    Parameters:
    -----------
    JD  ... Julian Day

    Returns:
    --------
    MJD ... Modified Julian Day
    """

    return JD-2400000.5

############################################
# Force / acceleration calculation
###########################################
def acc_ppn(t, y, GMarray=GMarray, Ephemlist=Ephemlist, Ephemorder=Ephemorder, ppnlist=ppnlist, au=au, clight=clight):
    """
    Parameterized Post Newtonian accelerations for massless body in the Solar System.

    Parameters:
    -----------
    t            ... Epochs
    y            ... State (positions and velocities)
    DART         ... Toggle for whether the DART impact and resulting acceleration is considered
    GMarray      ... Arrau of G*mass for all perturbing bodies
    Ephemlist    ... List linking the name of perturbers to their Ephemeris id
    Ephemorder   ... Order in which the accelerations should be summed. 
    ppnlist      ... List, which perturbers should have Post Newtonian terms added to their accelerations?
    au           ... Astronomical unit [km]
    clight       ... Speed of light [km/s]

    Returns:
    -------
    dydt         ... Derivative of the state vector (dx/dt, dv/dt)
    """
    n = len(Ephemorder)
    dydt = np.zeros(6)
    kmps2aupd = 86400.0 / au
    c = clight * kmps2aupd
    time = mjd2et(t)
    obj_states = array(spkez_mod(Ephemlist, time, 'ECLIPJ2000', 'NONE', 0))
    obj_pos = obj_states[:, 0:3] / au
    obj_vel = obj_states[:, 3:6] * kmps2aupd
    r_rel_mat = y[:3] - obj_pos
    v_rel_mat = y[3:6] - obj_vel
    dist_mat = np.linalg.norm(r_rel_mat, axis=1)
    acc = (GMarray / dist_mat**3)[:, None] * -r_rel_mat
    for i in range(n):
        obj = Ephemorder[i]
        if ppnlist[i]:
            acc[i, :] += acc_ppn_1order(r_rel_mat[i, :], dist_mat[i], v_rel_mat[i, :], c, GMarray[i])
        if obj == 'Sun':
            acc[i, :] += acc_J2(0, r_rel_mat[i, :], dist_mat[i], GMarray[i], au)
        elif obj == 'Earth':
            acc[i, :] += acc_J2(1, r_rel_mat[i, :], dist_mat[i], GMarray[i], au)
    dydt[:3] = y[3:6]
    dydt[3:6] = np.sum(acc, axis=0)
    return dydt

@jit('float64[:](float64[:], float64, float64[:], float64, float64)', nopython=True, cache=True)
def acc_ppn_1order(r_rel, dist, v_rel, c, GM):
    # sourcery skip: inline-immediately-returned-variable
    r_rel = ascontiguousarray(r_rel)
    v_rel = ascontiguousarray(v_rel)
    ppn_1order = GM/dist**3*((1/c**2)*((4.*GM/dist-np.dot(v_rel, v_rel))*r_rel + 4.*np.dot(r_rel, v_rel)*v_rel))
    return ppn_1order

@jit('float64[:](int64, float64[:], float64, float64, float64)', nopython=True, cache=True)
def acc_J2(idx, r_rel, dist, GM, au):
    """
    J2 zonal acceleration calculation of a specified celestial body

    Parameters:
    -----------
    obj          ... Object for which J2 acceleration is to be calculated
    obj_pos      ... Barycentric position of object
    y            ... State (positions and velocities)
    dist         ... 
    GMlist       ... List of G*mass for all perturbing bodies
    GMdict       ... Dictionary linking the name of perturbers to their GM id
    Ephemdict    ... Dictionary linking the name of perturbers to their Ephemeris id
    au           ... Astronomical unit [km]

    Returns:
    -------
    acc_J2       ... J2 acceleration of specified celestial body
    """
    ################################# DERIVATION START #################################
    # from sympy import symbols, simplify
    # from sympy.tensor.array import derive_by_array
    # GM, x, y, z, J2, radius = symbols('GM, x, y, z, J2, radius')
    # r = (x**2+y**2+z**2)**0.5
    # sin_phi = z/r
    # U_j2_JPL = -GM*J2/r**3*radius**2*0.5*(3*sin_phi**2 - 1) # https://ipnpr.jpl.nasa.gov/progress_report/42-196/196C.pdf, Equation 7
    # acc_J2 = simplify(derive_by_array(U_j2_JPL, (x, y, z)))
    ################################# DERIVATION END #################################

    radius_arr    = array((696000.0/au,           6378.1363/au)) # https://ipnpr.jpl.nasa.gov/progress_report/42-196/196C.pdf, Table 9
    J2_arr        = array((2.1106088532726840e-7, 0.00108262545)) # https://ipnpr.jpl.nasa.gov/progress_report/42-196/196C.pdf, Table 10
    obliquity_arr = array((7.25*np.pi/180,        84381.448/3600*np.pi/180))
    # sun obliquity, https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html, convert from degrees to radians
    # earty obliquity, https://ssd.jpl.nasa.gov/?constants, convert from seconds of arc to radians

    radius = radius_arr[idx]
    J2 = J2_arr[idx]
    obliq = obliquity_arr[idx]
    eclip2equat = array(([1,        0,                0    ],
                            [0,  cos(obliq), -sin(obliq)],
                            [0,  sin(obliq),  cos(obliq)])) # https://archive.org/details/131123ExplanatorySupplementAstronomicalAlmanac/page/n291/mode/2up\n,
    equat2eclip = array(([1,          0,              0    ],
                            [0,     cos(obliq),     sin(obliq)],
                            [0,    -sin(obliq),     cos(obliq)])) # https://archive.org/details/131123ExplanatorySupplementAstronomicalAlmanac/page/n291/mode/2up\n,
    x, y, z = ascontiguousarray(eclip2equat) @ ascontiguousarray(r_rel)
    acc_J2_equat  = array(( GM*J2*radius**2*x*(7.5*z**2 - 1.5*dist**2.0)*dist**(-7),
                            GM*J2*radius**2*y*(7.5*z**2 - 1.5*dist**2.0)*dist**(-7),
                        7.5*GM*J2*radius**2*z**3*dist**(-7) - 4.5*GM*J2*radius**2*z*dist**(-5) ))
    return ascontiguousarray(equat2eclip) @ ascontiguousarray(acc_J2_equat)


# Gauß–Radau integrator of order 15
# 
# This is a 15th order integrator that uses Gauß–Radau spacings to compute states 
# through timesteps that are determined adaptively. Additional information about the
# integrator and its implementation can be found at the following sources:
# 
# [1] Everhart, E. (1985). An efficient integrator that uses Gauss-Radau spacings. International Astronomical Union Colloquium, 83, 185-202. https://doi.org/10.1017/S0252921100083913
# [2] Rein, H., Spiegel, D.S. (2015). IAS15: a fast, adaptive, high-order integrator for gravitational dynamics, accurate to machine precision over a billion orbits. Monthly Notices of the Royal Astronomical Society, 446, 2, 1424–1437. https://doi.org/10.1093/mnras/stu2164
# [3] IAS15 Github (in C) https://github.com/hannorein/rebound/blob/master/src/integrator_ias15.c
# [4] Roa, J., Hamers, A. S., Cai, M. X., Leigh, N. W. (2020). Moving planets around: An introduction to n-body sinaif_body_GMlations applied to exoplanetary systems. The MIT Press. ISBN: 9780262539340. https://mitpress.ublish.com/ereader/9943/?preview=#page/Cover
# [5] MIT Book GitHub (in C and Python) https://github.com/MovingPlanetsAround/ABIE/blob/master/ABIE/integrator_gauss_radau15.py

"""
gauss_radau_15

A Gauß-Radau integrator of order 15
For a comprehensive guide, check Chapter 8 of [4] Moving Planets Around (listed above)
Implementation: Python 3.8, R. Makadia 20210108
"""
# START INTEGRATING!!!
def get_gr15_constants():
    '''
    Returns the constants needed for a 15th order Gauss-Radau Integrator

            Returns:
                    h (array): Gauss-Radau spacings of order 8
                    r (array): Coefficients to relate function evaluations and g matrix
                    c (array): Coefficients to relate g matrix and b matrix
    '''
    # from [3]
    h = array([0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648, 0.547153626330555383001448554766,	0.734210177215410531523210605558, 0.885320946839095768090359771030, 0.977520613561287501891174488626])
    num_nodes = h.size

    # from [5]
    r  =  array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [17.773808914078000840752659565672904106978971632681, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.5481367185372165056928216140765061758579336941398, 8.0659386483818866885371256689687154412267416180207, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.8358760786444386782520104428042437400879003147949, 3.3742499769626352599420358188267460448330087696743, 5.8010015592640614823286778893918880155743979164251, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.8276402675175978297946077587371204385651628457154, 2.0371118353585847827949159161566554921841792590404, 2.7254422118082262837742722003491334729711450288807, 5.1406241058109342286363199091504437929335189668304, 0.0, 0.0, 0.0, 0.0],
                [1.3620078160624694969370006292445650994197371928318, 1.4750402175604115479218482480167404024740127431358, 1.8051535801402512604391147435448679586574414080693, 2.6206449263870350811541816031933074696730227729812, 5.34597689987110751412149096322778980457703366603548, 0.0, 0.0, 0.0],
                [1.1295338753367899027322861542728593509768148769105, 1.2061876660584456166252036299646227791474203527801, 1.4182782637347391537713783674858328433713640692518, 1.8772424961868100972169920283109658335427446084411, 2.9571160172904557478071040204245556508352776929762, 6.6176620137024244874471284891193925737033291491748, 0.0, 0.0],
                [1.0229963298234867458386119071939636779024159134103, 1.0854721939386423840467243172568913862030118679827, 1.2542646222818777659905422465868249586862369725826, 1.6002665494908162609916716949161150366323259154408, 2.3235983002196942228325345451091668073608955835034, 4.1099757783445590862385761824068782144723082633980, 10.846026190236844684706431007823415424143683137181, 0.0]])

    # from [5]
    c  =  array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.562625605369221464656522e-1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [ 0.1014080283006362998648180399549641417413495311078e-1,  -0.2365032522738145114532321e0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.35758977292516175949344589284567187362040464593728e-2,  0.9353769525946206589574845561035371499343547051116e-1,  -0.5891279693869841488271399e0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [ 0.19565654099472210769005672379668610648179838140913e-2, -0.54755386889068686440808430671055022602028382584495e-1,  0.41588120008230686168862193041156933067050816537030e0, -0.11362815957175395318285885e1, 1.0, 0.0, 0.0, 0.0],
                [-0.14365302363708915424459554194153247134438571962198e-2,  0.42158527721268707707297347813203202980228135395858e-1, -0.36009959650205681228976647408968845289781580280782e0,  0.12501507118406910258505441186857527694077565516084e1, -0.18704917729329500633517991e1, 1.0, 0.0, 0.0],
                [ 0.12717903090268677492943117622964220889484666147501e-2, -0.38760357915906770369904626849901899108502158354383e-1,  0.36096224345284598322533983078129066420907893718190e0, -0.14668842084004269643701553461378480148761655599754e1,  0.29061362593084293014237914371173946705384212479246e1, -0.27558127197720458314421589e1, 1.0, 0.0],
                [-0.12432012432012432012432013849038719237133940238163e-2,  0.39160839160839160839160841227582657239289159887563e-1, -0.39160839160839160839160841545895262429018228668896e0,  0.17948717948717948717948719027866738711862551337629e1, -0.43076923076923076923076925231853900723503338586335e1,  0.56000000000000000000000001961129300233768803845526e1, -0.37333333333333333333333334e1, 1.0]])

    return h, r, c
h, r, c = get_gr15_constants()

@jit('Tuple((float64[:,:], float64[:,:]))(float64[:,:], float64[:,:], float64[:,:], float64[:,:], int64, float64[:,:])', nopython=True, cache=True)
def gr15_coefficients(g, b, r, c, h_idx, accels):
    '''
    Compute/refine the b and g matrices for approximating states

            Parameters:
                    g (array): Coefficients to refine b matrix at every Gauss-Radau spacing
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    r (array): Coefficients to relate function evaluations and g matrix
                    c (array): Coefficients to relate g matrix and b matrix
                    h_idx (int): Index of Gauss-Radau spacing being computed
                    accels (array): [au/day^2] Function values at each Gauss-Radau spacing

            Returns:
                    g (array): Coefficients to refine b matrix at every Gauss-Radau spacing
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
    '''
    F1 = accels[0, :]
    F2 = accels[1, :]
    F3 = accels[2, :]
    F4 = accels[3, :]
    F5 = accels[4, :]
    F6 = accels[5, :]
    F7 = accels[6, :]
    F8 = accels[7, :]

    if   h_idx == 1:
        g[0, :] = (F2 - F1) * r[1, 0]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
    elif h_idx == 2:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
    elif h_idx == 3:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
        b[2,:] =                               + c[2,2]*g[2,:] + c[3,2]*g[3,:] + c[4,2]*g[4,:] + c[5,2]*g[5,:] + c[6,2]*g[6,:]
    elif h_idx == 4:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
        g[3, :] = ((((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2] - g[2, :]) * r[4, 3]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
        b[2,:] =                               + c[2,2]*g[2,:] + c[3,2]*g[3,:] + c[4,2]*g[4,:] + c[5,2]*g[5,:] + c[6,2]*g[6,:]
        b[3,:] =                                                 c[3,3]*g[3,:] + c[4,3]*g[4,:] + c[5,3]*g[5,:] + c[6,3]*g[6,:]
    elif h_idx == 5:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
        g[3, :] = ((((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2] - g[2, :]) * r[4, 3]
        g[4, :] = (((((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2] - g[2, :]) * r[5, 3] - g[3, :]) * r[5, 4]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
        b[2,:] =                               + c[2,2]*g[2,:] + c[3,2]*g[3,:] + c[4,2]*g[4,:] + c[5,2]*g[5,:] + c[6,2]*g[6,:]
        b[3,:] =                                                 c[3,3]*g[3,:] + c[4,3]*g[4,:] + c[5,3]*g[5,:] + c[6,3]*g[6,:]
        b[4,:] =                                                                 c[4,4]*g[4,:] + c[5,4]*g[5,:] + c[6,4]*g[6,:]
    elif h_idx == 6:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
        g[3, :] = ((((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2] - g[2, :]) * r[4, 3]
        g[4, :] = (((((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2] - g[2, :]) * r[5, 3] - g[3, :]) * r[5, 4]
        g[5, :] = ((((((F7 - F1) * r[6, 0] - g[0, :]) * r[6, 1] - g[1, :]) * r[6, 2] - g[2, :]) * r[6, 3] - g[3, :]) * r[6, 4] - g[4, :]) * r[6, 5]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
        b[2,:] =                               + c[2,2]*g[2,:] + c[3,2]*g[3,:] + c[4,2]*g[4,:] + c[5,2]*g[5,:] + c[6,2]*g[6,:]
        b[3,:] =                                                 c[3,3]*g[3,:] + c[4,3]*g[4,:] + c[5,3]*g[5,:] + c[6,3]*g[6,:]
        b[4,:] =                                                                 c[4,4]*g[4,:] + c[5,4]*g[5,:] + c[6,4]*g[6,:]
        b[5,:] =                                                                                 c[5,5]*g[5,:] + c[6,5]*g[6,:]
    elif h_idx == 7:
        g[0, :] = (F2 - F1) * r[1, 0]
        g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
        g[3, :] = ((((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2] - g[2, :]) * r[4, 3]
        g[4, :] = (((((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2] - g[2, :]) * r[5, 3] - g[3, :]) * r[5, 4]
        g[5, :] = ((((((F7 - F1) * r[6, 0] - g[0, :]) * r[6, 1] - g[1, :]) * r[6, 2] - g[2, :]) * r[6, 3] - g[3, :]) * r[6, 4] - g[4, :]) * r[6, 5]
        g[6, :] = (((((((F8 - F1) * r[7, 0] - g[0, :]) * r[7, 1] - g[1, :]) * r[7, 2] - g[2, :]) * r[7, 3] - g[3, :]) * r[7, 4] - g[4, :]) * r[7, 5] - g[5, :]) * r[7, 6]

        b[0,:] = c[0,0]*g[0,:] + c[1,0]*g[1,:] + c[2,0]*g[2,:] + c[3,0]*g[3,:] + c[4,0]*g[4,:] + c[5,0]*g[5,:] + c[6,0]*g[6,:]
        b[1,:] =               + c[1,1]*g[1,:] + c[2,1]*g[2,:] + c[3,1]*g[3,:] + c[4,1]*g[4,:] + c[5,1]*g[5,:] + c[6,1]*g[6,:]
        b[2,:] =                               + c[2,2]*g[2,:] + c[3,2]*g[3,:] + c[4,2]*g[4,:] + c[5,2]*g[5,:] + c[6,2]*g[6,:]
        b[3,:] =                                                 c[3,3]*g[3,:] + c[4,3]*g[4,:] + c[5,3]*g[5,:] + c[6,3]*g[6,:]
        b[4,:] =                                                                 c[4,4]*g[4,:] + c[5,4]*g[5,:] + c[6,4]*g[6,:]
        b[5,:] =                                                                                 c[5,5]*g[5,:] + c[6,5]*g[6,:]
        b[6,:] =                                                                                                 c[6,6]*g[6,:]

    return g, b


def get_initial_timestep(t0_actual, state0_actual, F0):
    # sourcery skip: move-assign
    '''
    Compute the initial timestep and initialize the timestep counter

            Parameters:
                    t0_actual (float): [MJD], Epoch of integration period
                    state0_actual (array): [au, au/day], Didymos state at integration epoch
                    F0 (array): [au/day^2], Function evaluation at integration epoch

            Returns:
                    dt0 (float): [day], Initial integration timestep
                    timestep_counter (int): Timestep counter (1 because this function evaluates initial timestep)
    '''
    integrator_order = 15
    # Evaluate initial timestep
    pos0 = state0_actual[:3]
    vel0 = state0_actual[3:6]
    param0 = max(abs(pos0))
    param1 = max(abs(F0))
    if (param0 < 1e-5) or (param1 < 1e-5):
        test_dt0 = 1e-6
    else:
        test_dt0 = 0.01 * (param0 / param1)

    # Perform one Euler step
    pos1 = pos0 + test_dt0 * vel0
    vel1 = vel0 + test_dt0 * F0
    t1 = t0_actual + test_dt0
    state1 = concat([pos1, vel1])
    F1 = acc_ppn(t1, state1)[3:6]
    param2 = max(abs(F1 - F0)) / test_dt0
    if max(param1, param2) <= 1e-15:
        test_dt1 = max([1e-6, test_dt0 * 1e-3])
    else:
        test_dt1 = (0.01 / max(param1, param2)) ** (1.0 / (integrator_order + 1))

    dt0 = min(100*test_dt0, test_dt1)
    timestep_counter = 1

    return dt0, timestep_counter


@jit('float64[:](float64[:], float64[:], float64, float64[:,:], float64)', nopython=True, cache=True)
def approximate_state(state0, acc0, h_val, b, dt):
    '''
    Approximate the state at specified spacing given a b matrix and a timestep

            Parameters:
                    state0 (array): [au, au/day], Didymos state at timestep epoch
                    acc0 (array): [au/day^2], Function evaluation at timestep epoch
                    h_val (array): Gauss-Radau spacing to evaluate state at
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    dt (float): [day], Integration timestep corresponding to state0

            Returns:
                    approx_state (array): State approximation at specified Gauss-Radau spacing for the given timestep
    '''
    pos0 = state0[:3]
    vel0 = state0[3:6]

    approx_pos = pos0 + dt*h_val*(vel0 + dt*h_val*(acc0/2.0 + h_val*(b[0,:]/6.0 + h_val*(b[1,:]/12.0 + h_val*(b[2,:]/20.0 + h_val*(b[3,:]/30.0 + h_val*(b[4,:]/42.0 + h_val*(b[5,:]/56.0 + h_val*b[6,:]/72.0))))))))
    approx_vel = vel0 + dt*h_val*(acc0 + h_val*(b[0,:]/2.0 + h_val*(b[1,:]/3.0 + h_val*(b[2,:]/4.0 + h_val*(b[3,:]/5.0 + h_val*(b[4,:]/6.0 + h_val*(b[5,:]/7.0 + h_val*b[6,:]/8.0)))))))
    return concat((approx_pos, approx_vel))


@jit('Tuple((float64[:,:], float64[:,:], int64))(float64[:,:], float64, float64[:,:], int64)', nopython=True, cache=True)
def refine_b_matrix(b, q, e, timestep_counter):
    '''
    Refines the b matrix from end of one timestep for beginning of next timestep

            Parameters:
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    q (float): Ratio of required next timestep to current timestep
                    e (array): Auxiliary coefficient matrix from previous timestep
                    timestep_counter (int): Timestep counter

            Returns:
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    e (array): Auxiliary coefficient matrix for next timestep
                    timestep_counter (int): Timestep counter
    '''
    delta_b = b*0 if timestep_counter == 1 else b - e
    q2 = q*q
    q3 = q2*q
    q4 = q2*q2
    q5 = q2*q3
    q6 = q3*q3
    q7 = q3*q4

    e[0, :] = q  * (b[6, :] * 7.00 + b[5, :] * 6.00 + b[4, :] * 5.00 + b[3, :] * 4.0 + b[2, :] * 3.0 + b[1, :] * 2.0 + b[0, :])
    e[1, :] = q2 * (b[6, :] * 21.0 + b[5, :] * 15.0 + b[4, :] * 10.0 + b[3, :] * 6.0 + b[2, :] * 3.0 + b[1, :])
    e[2, :] = q3 * (b[6, :] * 35.0 + b[5, :] * 20.0 + b[4, :] * 10.0 + b[3, :] * 4.0 + b[2, :])
    e[3, :] = q4 * (b[6, :] * 35.0 + b[5, :] * 15.0 + b[4, :] * 5.00 + b[3, :])
    e[4, :] = q5 * (b[6, :] * 21.0 + b[5, :] * 6.00 + b[4, :])
    e[5, :] = q6 * (b[6, :] * 7.00 + b[5, :])
    e[6, :] = q7 * (b[6, :])

    b = e + delta_b
    timestep_counter += 1

    return b, e, timestep_counter

def gr15_step(t0_old, t0, state0, acc0, dt, dt_factor, dt_max, tf_actual, tol, h, r, c, g, b, b_old, e, adaptive_timestep, timestep_counter):
    # sourcery skip: low-code-quality
    '''
    Propagate the state through one timestep

            Parameters:
                    t0 (float): [MJD], Epoch for particular timestep
                    state0 (array): [au, au/day], Didymos state at timestep epoch
                    acc0 (array): [au/day^2], Function evaluation at timestep epoch
                    dt (float): [day], Integration timestep corresponding to state0
                    dt_factor (float): [day], Factor for maximum change in consecutive timesteps
                    dt_max (float): [day], Maximum timestep
                    tf_actual (float): [MJD], End time of integration period
                    tol (float): Integrator error tolerance
                    h (array): Gauss-Radau spacings of order 8
                    r (array): Coefficients to relate function evaluations and g matrix
                    c (array): Coefficients to relate g matrix and b matrix
                    g (array): Coefficients to refine b matrix at every Gauss-Radau spacing
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    b_old (array): Coefficients to refine state estimate at every Gauss-Radau spacing from previous timestep
                    e (array): Auxiliary coefficient matrix for current timestep
                    adaptive_timestep (boolean): Toggle for adaptive or fixed timestep
                    timestep_counter (int): Timestep counter

            Returns:
                    t0 (float): [MJD], Epoch for next timestep
                    state0_next (array): [au, au/day], Didymos state at next timestep epoch
                    acc0_next (array): [au, au/day^2], Function evaluation at next timestep epoch
                    dt (float): [MJD], Next integration timestep
                    g (array): Coefficients to refine b matrix at every Gauss-Radau spacing
                    b (array): Coefficients to refine state estimate at every Gauss-Radau spacing
                    b_old (array): Coefficients to refine state estimate at every Gauss-Radau spacing from previous timestep
                    e (array): Auxiliary coefficient matrix for next timestep
                    timestep_counter (int): Timestep counter
                    integrator_flag (int): Status of integrator step (0: initialized, 1: Step accepted, 2: Integration end reached)
    '''
    
    integrator_flag = 0
    max_pc_iter = 12
    pc_tol = 1e-16
    exp = 1.0/7.0 if adaptive_timestep else 0.0
    curr_iter = 0
    while True:
        curr_iter += 1
        # predictor-corrector loop
        for _ in range(max_pc_iter):
            temp_state = zeros((len(h),6))
            accels = zeros((len(h),3))
            temp_state[0, :] = state0
            accels[0, :] = acc0
            g, b = gr15_coefficients(g, b, r, c, 0, accels)
            for h_idx in range(1, len(h)):
                temp_state[h_idx, :] = approximate_state(state0, acc0, h[h_idx], b, dt)
                accels[h_idx, :] = acc_ppn(t0+h[h_idx]*dt, temp_state[h_idx, :])[3:6]
                g, b = gr15_coefficients(g, b, r, c, h_idx, accels)
            
            delta_b_tilde = b[-1,:] - b_old[-1,:]
            if timestep_counter > 2 and max(abs(delta_b_tilde))/max(abs(accels[-1,:])) < pc_tol:
                break
            b_old = b
        
        # calculate next timestep's state
        state0_next = approximate_state(state0, acc0, 1.0, b, dt)
        acc0_next = acc_ppn(t0+dt, state0_next)[3:6]
        b_tilde = max(abs(b[-1,:])) / max(abs(acc0_next))
        rel_error = (b_tilde/tol) ** exp
        dt_req = dt/rel_error

        # iteration acceptance if statement
        if rel_error <= 1:
            t0_old = t0
            t0 = t0 + dt
            b_old = b
            q = dt_req/dt
            b, e, timestep_counter = refine_b_matrix(b, q, e, timestep_counter)            
            integrator_flag = 2 if t0 == tf_actual else 1 # integrator has reached final time (2) or step accepted (1)
        
        if dt_req/dt > 1.0/dt_factor:
            dt = dt/dt_factor
        elif abs(dt_req) < 1e-12:
            dt = dt*dt_factor
        else:
            dt = dt_req
        if tf_actual > t0 and dt > dt_max or tf_actual < t0 and dt < dt_max:
            dt = dt_max

        if tf_actual > t0 and t0+dt > tf_actual  or  tf_actual < t0 and t0+dt < tf_actual:
            dt = tf_actual - t0
        
        if curr_iter >= 20: # abort integration if it gets stuck at one timestep
            integrator_flag = 2
        
        if integrator_flag > 0:
            break

    return t0_old, t0, state0_next, acc0_next, dt, g, b, b_old, e, timestep_counter, integrator_flag

# main propagator function
def propagate_gr15(t0, state0, tf, t_eval=array([]), adaptive_timestep=True, dt0=None, dt_max=6, tol=1e-6, dt_factor=0.25, NAIF_ID='399', ca_tol=0.1):
    # sourcery skip: low-code-quality
    '''
    Pass an initial state through an N-body propagator that uses the 15th order Gauss-Radau integrator

            Parameters:
                    t0 (float): [MJD], Epoch of integration period
                    state0 (array): [au, au/day], Didymos state at timestep epoch
                    tf (float): [MJD], End time of integration period
                    t_eval (array): [MJD], Epochs to return Didymos state at
                    adaptive_timestep (boolean): Toggle for adaptive or fixed timestep
                    dt0 (float): [day], Initial timestep
                    dt_max (float): [day], Maximum timestep
                    tol (float): Integrator error tolerance
                    dt_factor (float): [day], Factor for maximum change in consecutive timesteps
                    NAIF_ID (str): NAIF object ID of body to calculate close approaches for
                    ca_tol (float): [au], Close approach maximum distance tolerance

            Returns:
                    t0_array (array): [MJD], array of integration epochs
                    state0_array (array): [au, au/day], Didymos state at integration epochs
                    state_eval (array): [au, au/day], Didymos state at evaluation epochs
                    close_approach_summary (array): [boolean, MJD, (au, au/day), au], Didymos close approach data [impact boolean, time, relative state, b plane parameters]
    '''
    if t0 == tf:
        print('Final time is same as integration start time!')
        return t0, state0, None
    forward_prop = 1 if tf > t0 else 0
    b_old      = zeros((len(h)-1,3))
    b          = zeros((len(h)-1,3))
    g          = zeros((len(h)-1,3))
    e          = zeros((len(h)-1,3))

    acc0 = acc_ppn(t0, state0)[3:6]
    # use predefined initial timestep or calculate initial timestep
    if dt0 is not None:
        dt = dt0
        timestep_counter = 1
    else:
        dt, timestep_counter = get_initial_timestep(t0, state0, acc0)
    # if initial timestep is bigger than final time - forward integration
    if forward_prop and t0+dt>tf:
        dt = tf - t0
    # if integrating backwards in time, switch signs
    if not forward_prop:
        dt = -dt
        dt_max = -dt_max
        # if initial timestep is bigger than final time - backward integration
        if t0+dt<tf:
            dt = tf - t0

    t0_old = t0
    t0_list = [t0]
    state0_list = [state0]
    acc0_list = [acc0]
    b_list = []
    integrator_flag = 0 # means integrator is initializing
    while integrator_flag != 2:
        t0_old, t0, state0, acc0, dt, g, b, b_old, e, timestep_counter, integrator_flag = gr15_step(t0_old, t0, state0, acc0, dt, dt_factor, dt_max, tf, tol, h, r, c, g, b, b_old, e, adaptive_timestep, timestep_counter)
        t0_list.append(t0)
        state0_list.append(state0)
        acc0_list.append(acc0)
        b_list.append(b_old)

    t0_array = array(t0_list)
    state0_array = array(state0_list)
    acc0_array = array(acc0_list)
    dt_array = np.append(np.diff(t0_array), 0.0)
    b_array = array(b_list)

    if t_eval.size == 1:
        if tf > t0_array[0] and t_eval <= t0_array[-1]:
            state_eval = gr15_interpolate(np.float64(t_eval), t0_array, state0_array, acc0_array, dt_array, b_array, h)
    elif tf > t0_array[0]: # if t_eval is array in forward propagation
        state_eval = array([gr15_interpolate(t_interp, t0_array, state0_array, acc0_array, dt_array, b_array, h) for t_interp in t_eval if t_interp <= t0_array[-1]])
    else: # if t_eval is array in backward propagation
        state_eval = array([gr15_interpolate(t_interp, t0_array, state0_array, acc0_array, dt_array, b_array, h) for t_interp in t_eval if t_interp >= t0_array[-1]])

    return t0_array, state0_array, state_eval

@jit('float64[:,:](float64[:], float64[:,:])', nopython=True, cache=True)
def newton_coefficients(t_data, state_data):
    '''
    Calculate Newton interpolation coefficients given states at a set of time

            Parameters:
                    t_data (array): [MJD], array of integration epoch and corresponding Gauss-Radau spacings
                    state_data (array): [au, au/day], Didymos states at t_data
                    
            Returns:
                    coeffs (array): Array of interpolating coefficients
    '''
    n = len(t_data)
    state_size = state_data.shape[1]
    c = zeros((state_size, n, n))
    for state_idx in range(state_size):
        c[state_idx, :, 0] = state_data[:, state_idx]
        for j in range(1, n):
            for k in range(n-j):
                c[state_idx, k, j] = (c[state_idx, k+1, j-1] - c[state_idx, k, j-1])/(t_data[k+j]-t_data[k])
    
    return c[:, 0, :]

@jit('float64[:](float64[:,:], float64[:], float64)', nopython=True, cache=True)
def newton_polynomial(coeffs, t_data, t_interp):
    '''
    Evaluate Newton polynomial to find integrated state at an interpolation time

            Parameters:
                    coeffs (array): Array of interpolating coefficients
                    t_data (array): [MJD], Array of integration epoch and corresponding Gauss-Radau spacings
                    t_interp (float): [MJD], Date at which state needs to be interpolated
                    
            Returns:
                    state_interp (array): [au, au/day], Interpolated state at t_interp     
    '''
    n = len(t_data)-1
    state_interp = coeffs[:, n]
    for i in range(1, n+1):
        state_interp = coeffs[:, n-i] + (t_interp-t_data[n-i])*state_interp

    return state_interp

@jit('int64(float64[:], float64)', nopython=True, cache=True)
def find_exact_or_previous_index(array, value):
    '''
    Find the index of the equal or previous entry to a value in the integrator result's time array

            Parameters:
                    array (array): Array in which to search for value
                    value (float): Value of integration epoch and corresponding Gauss-Radau spacings
                    
            Returns:
                    state_interp (array): [au, au/day], Interpolated state at t_interp     
    '''
    # find closest value index
    idx = (np.abs(array - value)).argmin()
    # if forward integration and array entry is bigger than value 
    if array[1]-array[0] > 0 and array[idx] > value:
        idx = idx - 1
    # if backward integration and array entry is smaller than value
    elif array[1]-array[0] < 0 and array[idx] < value:
        idx = idx - 1
    
    return idx

@jit('float64[:](float64, float64[:], float64[:,:], float64[:,:], float64[:], float64[:,:,:], float64[:])', nopython=True, cache=True)
def gr15_interpolate(t_interp, t0_array, state0_array, acc0_array, dt_array, b_array, h):
    '''
    Interpolate epochs and states generated by the Gauss-Radau integrator

            Parameters:
                    t_interp (float): [MJD], Date at which state needs to be interpolated
                    t0_array (array): [MJD], array of integration epochs
                    state0_array (array): [au, au/day], Didymos state at integration epochs
                    acc0_array (array): [au/day^2], Function values at integration epochs
                    dt_array (array): [day], Integration timestep array
                    b_array (array): 3-D array of coefficients used to refine state estimate at every Gauss-Radau spacing at every timestep
                    h (array): Gauss-Radau spacings of order 8
                    
            Returns:
                    state_interp (array): [au, au/day], Interpolated state at t_interp     
    '''
    idx = find_exact_or_previous_index(t0_array, t_interp)
    if t_interp == t0_array[idx]:
        return state0_array[idx, :]
    dt = dt_array[idx]
    t_data = t0_array[idx] + h*dt
    state_data = zeros((len(h),6))
    state_data[0, :] = state0_array[idx]
    for h_idx in range(1, len(h)):
        state_data[h_idx, :] = approximate_state(state0_array[idx], acc0_array[idx], h[h_idx], b_array[idx], dt)
    coeffs = newton_coefficients(t_data, state_data)

    return newton_polynomial(coeffs, t_data, t_interp)
