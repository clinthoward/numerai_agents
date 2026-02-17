from agents.code.signals.variants import DEFAULT_VARIANTS, DEFAULT_VARIANT_LADDER

VARIANTS = DEFAULT_VARIANTS
VARIANT_LADDER = DEFAULT_VARIANT_LADDER

VARIANT_LIST = [
    {"name": name, **body} for name, body in VARIANTS.items()
]

CONFIG = {
    "signals": {
        "variants": VARIANT_LADDER,
        "variant_definitions": VARIANTS,
        "variant": VARIANT_LIST,
        "benchmark": {
            "reference_variant": "v00_lgbm_baseline",
        },
    },
}
