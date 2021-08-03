from os import environ


BUILD_VERSION = environ.get('MLMODULE_BUILD_VERSION', '0.0.dev0')
