import re

from mozuma.cli.types import ArgMoZuMaOptions, CLICommandDefinition


def check_perf_fun(options: ArgMoZuMaOptions) -> None:
    """Check current setup for recommended performances."""
    from PIL import Image, features

    print("Running performance checks.")

    # libjpeg_turbo check
    # see https://fastai1.fast.ai/performance.html#libjpeg-turbo
    print("*** libjpeg-turbo status")
    if features.check_feature("libjpeg_turbo"):
        print("✔ libjpeg-turbo is on")
    else:
        print("✘ libjpeg-turbo is not on")

    # Pillow-SIMD check
    # see https://fastai1.fast.ai/performance.html#pillow-simd
    print("*** Pillow-SIMD status")
    pil_version = Image.__version__
    if re.search(r"\.post\d+", pil_version):
        print(f"✔ Running Pillow-SIMD {pil_version}")
    else:
        print(f"✘ Running Pillow {pil_version}")


COMMAND = CLICommandDefinition(
    name="check",
    help_text="Performance checks of current installation",
    args_parser=lambda x: None,
    command_fun=check_perf_fun,
    options_class=ArgMoZuMaOptions,
)
