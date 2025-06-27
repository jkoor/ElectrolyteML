from models.battery import Electrolyte
from source.SALT import LIPF6
from source.SOLVENT import DMC, EC
from models.materials import Salt, Solvent, Additive

e1 = Electrolyte.create(
    name="EC/DMC (1:1)",
    id="EC/DMC",
    description="A common electrolyte mixture used in lithium-ion batteries.",
    salts=[(LIPF6, 15)],
    solvents=[
        (EC, 50),
        (DMC, 50),
    ],
    additives=[],
    performance={},
)

e2 = Electrolyte.create(
    name="EC/DMC (1:1)",
    id="EC/DMC",
    description="A common electrolyte mixture used in lithium-ion batteries.",
    salts=[(LIPF6, 15)],
    solvents=[
        (EC, 30),
        (DMC, 70),
    ],
    additives=[],
    performance={},
)

# Electrolyte.from_jsons("database/electrolyte.json")

pass
