from battery import Electrolyte
from database.LITHIUMSALT import LIPF6
from database.SOLVENT import DMC, EC

e1 = Electrolyte.create(
    name="EC/DMC (1:1)",
    id="EC/DMC",
    description="A common electrolyte mixture used in lithium-ion batteries.",
    lithium_salts=[(LIPF6, 15)],
    solvents=[
        (EC, 50),
        (DMC, 50),
    ],
    additives=[],
    performance={},
)

pass
