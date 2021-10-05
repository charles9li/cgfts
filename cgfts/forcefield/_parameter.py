from __future__ import absolute_import, division, print_function

__all__ = ['_Parameter']


class _Parameter(object):

    def __init__(self, name, value, potential, fixed, use_value_func=False):
        self._name = name
        self._value = value
        self._potential = potential
        self._fixed = fixed
        self._use_value_func = use_value_func

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        if self._use_value_func:
            return self._value(self._potential)
        else:
            return self._value

    @value.setter
    def value(self, val):
        if self._use_value_func:
            error_message = "cannot set value of '{}' parameter because a " \
                            "function is used to return value".format(self._name)
            raise AttributeError(error_message)
        else:
            self._value = val

    @property
    def potential(self):
        return self._potential

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        if not isinstance(value, bool):
            raise ValueError("'fixed' attribute must be of type bool")
        self._fixed = value
