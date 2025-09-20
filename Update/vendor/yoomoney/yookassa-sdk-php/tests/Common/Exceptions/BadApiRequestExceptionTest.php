<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\BadApiRequestException;

/**
 * @internal
 */
class BadApiRequestExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new BadApiRequestException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode()
    {
        return BadApiRequestException::HTTP_CODE;
    }
}
