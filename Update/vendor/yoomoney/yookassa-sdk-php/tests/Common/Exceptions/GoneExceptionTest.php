<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\GoneException;

/**
 * @internal
 */
class GoneExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = ''): GoneException
    {
        return new GoneException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode(): int
    {
        return GoneException::HTTP_CODE;
    }
}
