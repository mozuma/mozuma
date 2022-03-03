def sanitize_resnet_arch(resnet_arch: str) -> str:
    return resnet_arch.lower().replace("_", "-")
