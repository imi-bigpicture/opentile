#    Copyright 2026 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Exceptions and warnings raised by opentile_jpeg."""


class JpegError(RuntimeError):
    """libjpeg-turbo rejected a JPEG; the message is libjpeg-turbo's own."""


class JpegTooLargeError(JpegError):
    """A JPEG claims more pixels than the operation's ``max_pixels`` allows.

    Raised before anything is allocated for the image, which a small corrupt or
    malicious file could otherwise make huge. Pass a larger ``max_pixels``, or
    None, to the call that reads a legitimately larger image.
    """


class JpegWarning(UserWarning):
    """libjpeg-turbo recovered from a problem in a JPEG, e.g. corrupt data.

    The operation still produced a result, but it may not be what the input
    intended. Escalate with ``warnings.simplefilter("error", JpegWarning)``.
    """
