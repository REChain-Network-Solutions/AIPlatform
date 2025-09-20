<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\TooManyRequestsException;

/**
 * @internal
 */
class TooManyRequestsExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new TooManyRequestsException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode()
    {
        return TooManyRequestsException::HTTP_CODE;
    }
}
