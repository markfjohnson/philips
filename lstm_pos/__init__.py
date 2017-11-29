#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pkg_resources

__author__ = "vinod kumar(vinod.kumar.s@philips.com)"
__copyright__ = "See LICENSE for details"
__license__ = "See LICENSE for details."

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
