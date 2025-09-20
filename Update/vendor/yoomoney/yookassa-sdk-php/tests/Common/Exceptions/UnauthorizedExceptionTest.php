<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\UnauthorizedException;

/**
 * @internal
 */
class UnauthorizedExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new UnauthorizedException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode()
    {
        return UnauthorizedException::HTTP_CODE;
    }
}
