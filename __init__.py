from .bfl_api import (
    FluxPro,
    FluxDev,
    FluxPro11,
    FluxCanny,
    FluxDepth,
    FluxUltra11,
    FluxProFill,
    FluxProFinetune,
    FluxProCannyFinetune,
    FluxProDepthFinetune,
    FluxProFillFinetune,
    FluxUltra11Finetune,
    FluxKontextProEdit,
    FluxKontextProT2I,
    FluxKontextMaxEdit,
    FluxKontextMaxT2I,
)

NODE_CLASS_MAPPINGS = {
    "FLUX 1.0 [pro]": FluxPro,
    "FLUX 1.0 [dev]": FluxDev,
    "FLUX 1.1 [pro]": FluxPro11,
    "FLUX 1.1 [ultra]": FluxUltra11,
    "FLUX 1.0 [depth]": FluxDepth,
    "FLUX 1.0 [canny]": FluxCanny,
    "FLUX 1.0 [fill]": FluxProFill,
    "FLUX 1.0 [pro] Finetuned": FluxProFinetune,
    "FLUX 1.0 [canny] Finetuned": FluxProCannyFinetune,
    "FLUX 1.0 [depth] Finetuned": FluxProDepthFinetune,
    "FLUX 1.0 [fill] Finetuned": FluxProFillFinetune,
    "FLUX 1.1 [ultra] Finetuned": FluxUltra11Finetune,
    "FLUX.1 Kontext [pro] Image Edit": FluxKontextProEdit,
    "FLUX.1 Kontext [pro] Text to Image": FluxKontextProT2I,
    "FLUX.1 Kontext [max] Image Edit": FluxKontextMaxEdit,
    "FLUX.1 Kontext [max] Text to Image": FluxKontextMaxT2I,
}
