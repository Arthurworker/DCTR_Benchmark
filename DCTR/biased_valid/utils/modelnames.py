from models.base_ctr import basectr
from models.ips_ctr import ipsctr
from models.idips_ctr import idipsctr
from models.udips_ctr import udipsctr
from models.cause_ctr import causectr
from models.bridge_ctr import bridgectr
from models.refine_ctr import refinectr
from models.tjr_ctr import tjrctr
from models.dr_ctr import drctr
from models.drjl_ctr import drjlctr

models = {
    "BASE-CTR": basectr,
    "IPS-CTR": ipsctr,
    "IDIPS-CTR": idipsctr,
    "UDIPS-CTR": udipsctr,
    "CausE-CTR": causectr,
    "Bridge-CTR": bridgectr,
    "Refine-CTR": refinectr,
    "TJR-CTR": tjrctr,
    "DR-CTR": drctr,
    "DRJL-CTR": drjlctr,
}

